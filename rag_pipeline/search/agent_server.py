from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .engine import (
    build_arg_parser,
    get_bge_m3_helper,
    get_qwen_embedder,
    preflight_runtime_checks,
)
from ..agents.brain_agent import brain_namespace_to_rag_overrides, run_brain_agent
from ..logging_utils import configure_pipeline_logging
from ..runtime_cache import TTLCache, make_cache_key


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Agent</title>
  <style>
    :root { color-scheme: light; --ink:#17202a; --muted:#64748b; --line:#d7dde6; --soft:#f6f8fb; --accent:#0f766e; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Microsoft YaHei", "Segoe UI", sans-serif; color: var(--ink); background: #fbfcfd; }
    main { max-width: 980px; margin: 0 auto; padding: 32px 20px 48px; }
    h1 { margin: 0 0 16px; font-size: 28px; font-weight: 700; letter-spacing: 0; }
    .bar { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: start; }
    textarea { width: 100%; min-height: 110px; resize: vertical; border: 1px solid var(--line); border-radius: 6px; padding: 12px; font: inherit; line-height: 1.6; background: white; }
    button { height: 44px; border: 0; border-radius: 6px; padding: 0 18px; font: inherit; font-weight: 700; color: white; background: var(--accent); cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    .meta { margin: 12px 0 18px; color: var(--muted); font-size: 14px; }
    pre { white-space: pre-wrap; word-break: break-word; border: 1px solid var(--line); border-radius: 6px; padding: 16px; min-height: 220px; line-height: 1.65; background: white; }
  </style>
</head>
<body>
  <main>
    <h1>RAG Agent</h1>
    <div class="bar">
      <textarea id="query" placeholder="输入问题，例如：可可资本 简介"></textarea>
      <button id="ask">提问</button>
    </div>
    <div id="meta" class="meta">服务已启动，模型首次预热后会更快。</div>
    <pre id="answer"></pre>
  </main>
  <script>
    const query = document.getElementById("query");
    const ask = document.getElementById("ask");
    const answer = document.getElementById("answer");
    const meta = document.getElementById("meta");
    async function runAsk() {
      const text = query.value.trim();
      if (!text) return;
      ask.disabled = true;
      answer.textContent = "";
      meta.textContent = "正在生成答案...";
      const started = performance.now();
      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query: text, answer_only: true, show_evidence: true})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        answer.textContent = data.answer_text || "";
        meta.textContent = `完成，用时 ${((performance.now() - started) / 1000).toFixed(1)} 秒`;
      } catch (err) {
        answer.textContent = String(err);
        meta.textContent = "请求失败";
      } finally {
        ask.disabled = false;
      }
    }
    ask.addEventListener("click", runAsk);
    query.addEventListener("keydown", event => {
      if (event.ctrlKey && event.key === "Enter") runAsk();
    });
  </script>
</body>
</html>
"""


class AgentRuntime:
    def __init__(self, base_args: argparse.Namespace):
        self.base_args = base_args
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.request_count = 0
        self.response_cache = TTLCache(
            ttl_seconds=int(os.getenv("RAG_AGENT_RESPONSE_CACHE_TTL_SECONDS", "900") or "0"),
            max_items=int(os.getenv("RAG_AGENT_RESPONSE_CACHE_MAX_ITEMS", "64") or "0"),
        )
        self.cache_hit_count = 0

    def _base_args_cache_payload(self, args: argparse.Namespace) -> Dict[str, Any]:
        ignored = {"query", "query_text", "json", "session_id"}
        payload: Dict[str, Any] = {}
        for key, value in sorted(vars(args).items()):
            normalized_key = str(key).lower()
            if key in ignored or "api_key" in normalized_key:
                continue
            payload[key] = value
        return payload

    def _response_cache_allowed(self, payload: Dict[str, Any], args: argparse.Namespace) -> bool:
        if not os.getenv("RAG_AGENT_RESPONSE_CACHE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}:
            return False
        if payload.get("raw"):
            return False
        if str(getattr(args, "session_id", "") or "").strip():
            return False
        return True

    def _response_cache_key(self, query: str, payload: Dict[str, Any], args: argparse.Namespace) -> str:
        return make_cache_key(
            "agent_server_response",
            {
                "query": query,
                "route": str(payload.get("route") or "auto").strip() or "auto",
                "answer_only": bool(payload.get("answer_only", True)),
                "show_evidence": bool(payload.get("show_evidence", True)),
                "evidence_top_k": payload.get("evidence_top_k"),
                "args": self._base_args_cache_payload(args),
            },
        )

    def warmup(self) -> None:
        preflight_runtime_checks(
            self.base_args.model_path,
            self.base_args.device,
            bge_m3_model_path=self.base_args.bge_m3_model_path,
            require_bge_m3=bool(
                (self.base_args.enable_bge_dense_retrieval or self.base_args.enable_bge_sparse_retrieval)
                and str(self.base_args.bge_m3_model_path or "").strip()
            ),
        )
        get_qwen_embedder(
            model_path=self.base_args.model_path,
            device=self.base_args.device,
            dtype=self.base_args.dtype,
            attn_implementation=self.base_args.attn_implementation,
            max_length=self.base_args.max_length,
            keep_loaded=True,
        )
        if (
            (self.base_args.enable_bge_dense_retrieval or self.base_args.enable_bge_sparse_retrieval)
            and str(self.base_args.bge_m3_model_path or "").strip()
        ):
            get_bge_m3_helper(
                model_path=self.base_args.bge_m3_model_path,
                device=self.base_args.bge_m3_device,
                batch_size=self.base_args.bge_m3_batch_size,
                query_max_length=self.base_args.bge_m3_query_max_length,
                passage_max_length=self.base_args.bge_m3_passage_max_length,
                use_fp16=self.base_args.bge_m3_use_fp16,
                keep_loaded=True,
            )

    def ask(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query") or "").strip()
        if not query:
            raise ValueError("query cannot be empty")

        args = copy.copy(self.base_args)
        args.query = [query]
        args.query_text = None
        args.answer_only = bool(payload.get("answer_only", True))
        args.show_evidence = bool(payload.get("show_evidence", True))
        args.json = bool(payload.get("raw", False))
        args.session_id = str(payload.get("session_id") or args.session_id or "").strip()
        if "evidence_top_k" in payload:
            args.answer_evidence_top_k = max(1, int(payload.get("evidence_top_k") or args.answer_evidence_top_k))

        started = time.perf_counter()
        cache_key = ""
        if self._response_cache_allowed(payload, args):
            cache_key = self._response_cache_key(query, payload, args)
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                self.cache_hit_count += 1
                cached_response["elapsed_seconds"] = round(time.perf_counter() - started, 2)
                cached_response["cache_hit"] = "agent_server_response"
                return cached_response

        with self.lock:
            self.request_count += 1
            agent_state = run_brain_agent(
                query,
                session_id=str(args.session_id or "").strip(),
                route=str(payload.get("route") or "auto").strip() or "auto",
                args_overrides=brain_namespace_to_rag_overrides(args),
            )
        output = dict(agent_state.get("raw_output") or {})
        answer_text = str(agent_state.get("answer_text") or "").strip()
        agent_errors = [str(item) for item in agent_state.get("errors") or []]
        if agent_errors and not answer_text:
            raise RuntimeError("; ".join(agent_errors))
        response = {
            "answer_text": answer_text,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "timings": ((output.get("local_state") or {}).get("timings") or {}),
            "trace_file": str((output.get("local_state") or {}).get("trace_file") or ""),
            "route": output.get("route", ""),
            "route_reason": output.get("route_reason", ""),
            "agent_trace": output.get("agent_trace", []),
            "grounding_mode": (output.get("merge") or {}).get("source", ""),
            "llm_model": (output.get("merge") or {}).get("model", ""),
            "agent": {
                "name": "brain_agent",
                "framework": "langgraph",
                "metadata": agent_state.get("metadata", {}),
            },
            "warnings": agent_errors,
        }
        if payload.get("raw"):
            response["raw"] = output
            response["agent_state"] = agent_state
        if cache_key and str(response.get("route") or "").strip().lower() == "local":
            self.response_cache.set(cache_key, response)
        return response


def make_handler(runtime: AgentRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RAGAgent/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[RAG_AGENT] {self.address_string()} - {fmt % args}")

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(
                    {
                        "ok": True,
                        "uptime_seconds": round(time.time() - runtime.started_at, 1),
                        "request_count": runtime.request_count,
                        "cache_hit_count": runtime.cache_hit_count,
                        "response_cache": runtime.response_cache.stats(),
                    }
                )
                return
            if parsed.path == "/ask":
                query = str(parse_qs(parsed.query).get("query", [""])[0]).strip()
                try:
                    self._send_json(runtime.ask({"query": query, "answer_only": True, "show_evidence": True}))
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if parsed.path in {"/", "/index.html"}:
                raw = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/ask":
                self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                self._send_json(runtime.ask(self._read_json()))
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default=os.getenv("RAG_AGENT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("RAG_AGENT_PORT", "7860")))
    parser.add_argument("--no-warmup", action="store_true", help="Skip loading local models at server startup.")
    return parser


def main() -> int:
    configure_pipeline_logging()
    server_parser = build_server_parser()
    server_args, engine_argv = server_parser.parse_known_args()
    engine_args = build_arg_parser().parse_args(engine_argv)
    engine_args.no_embedder_cache = False
    engine_args.answer_only = True
    engine_args.show_evidence = True
    engine_args.enable_llm_synthesis = True

    runtime = AgentRuntime(engine_args)
    if not server_args.no_warmup:
        print("[RAG_AGENT] Warming up local embedding models...")
        started = time.perf_counter()
        runtime.warmup()
        print(f"[RAG_AGENT] Warmup done in {time.perf_counter() - started:.1f}s")

    server = ThreadingHTTPServer((server_args.host, server_args.port), make_handler(runtime))
    print(f"[RAG_AGENT] Serving at http://{server_args.host}:{server_args.port}")
    print("[RAG_AGENT] POST /ask with JSON: {\"query\":\"...\"}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[RAG_AGENT] Stopping...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
