from __future__ import annotations

import argparse
import json
import os
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .web_analysis_agent import call_iqs_readpage, infer_search_options, iqs_api_key_is_configured, run_iqs_optimized_search


SERVER_NAME = "IQSWebSearchAPI"


class WebSearchRuntime:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.request_count = 0

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "service": "iqs_web_search",
            "provider": "aliyun_iqs_skills",
            "api_key_configured": iqs_api_key_is_configured(),
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "request_count": self.request_count,
        }

    def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query") or "").strip()
        if not query:
            raise ValueError("query 不能为空")
        if not iqs_api_key_is_configured():
            raise RuntimeError("ALIYUN_IQS_API_KEY 未配置。请把它写入项目根目录的 .env。")

        options = infer_search_options(query, payload)
        started = time.perf_counter()
        optimized = run_iqs_optimized_search(query, options)
        results = list(optimized.get("results") or [])
        self.request_count += 1
        return {
            "ok": True,
            "service": "iqs_web_search",
            "query": query,
            "options": options,
            "query_plan": optimized.get("query_plan", []),
            "search_tasks": optimized.get("search_tasks", []),
            "search_trace": optimized.get("search_trace", []),
            "quality_processing": optimized.get("quality_processing", {}),
            "warnings": optimized.get("errors", []),
            "count": len(results),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "results": results,
        }

    def readpage(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = str(payload.get("url") or "").strip()
        if not url:
            raise ValueError("url 不能为空")
        if not iqs_api_key_is_configured():
            raise RuntimeError("ALIYUN_IQS_API_KEY 未配置。请把它写入项目根目录的 .env。")
        timeout_ms = int(payload.get("timeout") or payload.get("timeout_ms") or os.getenv("IQS_SEARCH_TIMEOUT_MS", "60000"))

        started = time.perf_counter()
        result = call_iqs_readpage(url, timeout_ms=max(1000, min(180000, timeout_ms)))
        self.request_count += 1
        return {
            "ok": True,
            "service": "iqs_web_search",
            "url": url,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "result": result,
        }


def _query_payload(parsed_query: str) -> Dict[str, Any]:
    values = parse_qs(parsed_query)
    payload: Dict[str, Any] = {}
    for key, items in values.items():
        if not items:
            continue
        value = items[-1]
        if key in {"numResults", "num_results", "timeout", "timeout_ms"}:
            try:
                payload[key] = int(value)
            except ValueError:
                payload[key] = value
        else:
            payload[key] = value
    return payload


def make_handler(runtime: WebSearchRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"{SERVER_NAME}/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[IQS_SEARCH] {self.address_string()} - {fmt % args}")

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/health":
                    self._send_json(runtime.health())
                    return
                if parsed.path == "/search":
                    self._send_json(runtime.search(_query_payload(parsed.query)))
                    return
                if parsed.path == "/readpage":
                    self._send_json(runtime.readpage(_query_payload(parsed.query)))
                    return
                self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/search":
                    self._send_json(runtime.search(payload))
                    return
                if parsed.path == "/readpage":
                    self._send_json(runtime.readpage(payload))
                    return
                self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    return Handler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阿里云 IQS 联网搜索独立 HTTP 接口。")
    parser.add_argument("--host", default=os.getenv("IQS_SEARCH_API_HOST", "127.0.0.1"), help="监听地址。")
    parser.add_argument("--port", type=int, default=int(os.getenv("IQS_SEARCH_API_PORT", "7870")), help="监听端口。")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runtime = WebSearchRuntime()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"[IQS_SEARCH] Serving at http://{args.host}:{args.port}")
    print("[IQS_SEARCH] GET  /health")
    print("[IQS_SEARCH] POST /search   JSON: {\"query\":\"...\"}")
    print("[IQS_SEARCH] POST /readpage JSON: {\"url\":\"https://...\"}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[IQS_SEARCH] Stopping...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
