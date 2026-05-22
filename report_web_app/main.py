from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
RUN_FULL_REPORT = PIPELINE_ROOT / "run_full_report.py"
OUTPUT_ROOT = PIPELINE_ROOT / "output"
JOB_OUTPUT_ROOT = OUTPUT_ROOT / "web_reports"
STATIC_DIR = Path(__file__).resolve().parent / "static"
PYTHON_EXE = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_REPORT_ROUTE = "web"


class ReportRequest(BaseModel):
    main_title: str = Field(..., min_length=1, max_length=300)
    research_direction: str = Field(default="", max_length=2000)
    llm_profile: str = Field(default="", max_length=80)
    skip_reformatter: bool = False


@dataclass
class ReportJob:
    id: str
    main_title: str
    research_direction: str
    query: str
    route: str
    llm_profile: str
    skip_reformatter: bool
    output_dir: Path
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    command: List[str] = field(default_factory=list)
    error: str = ""
    log_path: Path = Path()
    log_tail: Deque[str] = field(default_factory=lambda: deque(maxlen=500))
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    delivery_status: Dict[str, Any] = field(default_factory=dict)
    process: Optional[subprocess.Popen[str]] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, ReportJob] = {}
        self._lock = threading.RLock()

    def add(self, job: ReportJob) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> ReportJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    def list_recent(self, limit: int = 20) -> List[ReportJob]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
        return jobs[:limit]

    def update(self, job: ReportJob, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(job, key, value)


jobs = JobStore()
app = FastAPI(title="Report Web App", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def compose_report_query(main_title: str, research_direction: str = "") -> str:
    title = " ".join(str(main_title or "").split())
    direction = str(research_direction or "").strip()
    if direction:
        return f"主标题：{title}\n研究方向：{direction}"
    return title


def _now_iso(value: Optional[float]) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def _file_info(job_id: str, kind: str, path: Path, label: str) -> Dict[str, Any]:
    path = path.resolve()
    exists = path.exists() and path.is_file()
    info: Dict[str, Any] = {
        "kind": kind,
        "label": label,
        "filename": path.name,
        "path": str(path),
        "exists": exists,
        "size": path.stat().st_size if exists else 0,
        "modified_at": _now_iso(path.stat().st_mtime) if exists else "",
        "download_url": f"/api/reports/{job_id}/files/{kind}" if exists else "",
    }
    if exists and path.suffix.lower() in {".md", ".txt", ".log"}:
        info["preview_url"] = f"/api/reports/{job_id}/content/{kind}"
    else:
        info["preview_url"] = ""
    return info


def _latest_file(output_dir: Path, pattern: str) -> Optional[Path]:
    matches = [item for item in output_dir.glob(pattern) if item.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_artifact_path(job: ReportJob, value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (PIPELINE_ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(job.output_dir.resolve())
    except ValueError:
        return None
    return path if path.exists() and path.is_file() else None


def refresh_artifacts(job: ReportJob) -> None:
    package_path = _latest_file(job.output_dir, "*.writer_package.json")
    package = _read_json(package_path) if package_path else {}
    writer_report = package.get("writer_report") if isinstance(package.get("writer_report"), dict) else {}
    delivery = package.get("report_delivery_status") if isinstance(package.get("report_delivery_status"), dict) else {}
    reformatter = package.get("reformatter_result") if isinstance(package.get("reformatter_result"), dict) else {}

    review_path = (
        _safe_artifact_path(job, delivery.get("review_draft_path"))
        or _safe_artifact_path(job, writer_report.get("review_draft_markdown_path"))
        or _latest_file(job.output_dir, "*_review_draft.md")
    )
    clean_candidate = (
        _safe_artifact_path(job, delivery.get("clean_report_path"))
        or _safe_artifact_path(job, writer_report.get("reformatter_output_path"))
        or _safe_artifact_path(job, reformatter.get("output_path"))
        or _latest_file(job.output_dir, "*_clean.md")
    )
    clean_path = None
    if package:
        if delivery.get("clean_report_written") and not delivery.get("review_required"):
            clean_path = clean_candidate
    else:
        clean_path = clean_candidate
    writer_path = _safe_artifact_path(job, writer_report.get("writer_markdown_path")) or _latest_file(job.output_dir, "*.writer.md")
    fallback_path = (
        _safe_artifact_path(job, delivery.get("fallback_report_path"))
        or _safe_artifact_path(job, writer_report.get("fallback_output_path"))
        or _safe_artifact_path(job, reformatter.get("fallback_output_path"))
        or _latest_file(job.output_dir, "*_fallback_writer.md")
    )
    diagnostic_path = (
        _safe_artifact_path(job, delivery.get("diagnostic_markdown_path"))
        or _safe_artifact_path(job, writer_report.get("diagnostic_markdown_path"))
        or _latest_file(job.output_dir, "*.diagnostic.md")
    )
    state_path = _latest_file(job.output_dir, "*.state.json")

    artifacts: Dict[str, Dict[str, Any]] = {}
    if review_path:
        artifacts["review_draft"] = _file_info(job.id, "review_draft", review_path, "待复核草稿")
    if clean_path and clean_path.exists() and clean_path.is_file():
        artifacts["final"] = _file_info(job.id, "final", clean_path, "最终通过报告")
    elif writer_path and not delivery.get("review_required") and job.returncode == 0:
        artifacts["final"] = _file_info(job.id, "final", writer_path, "最终通过报告")
    if writer_path:
        artifacts["writer"] = _file_info(job.id, "writer", writer_path, "Writer 原始报告")
    if fallback_path:
        artifacts["fallback"] = _file_info(job.id, "fallback", fallback_path, "回退报告")
    if diagnostic_path:
        artifacts["diagnostic"] = _file_info(job.id, "diagnostic", diagnostic_path, "诊断报告")
    if package_path:
        artifacts["package"] = _file_info(job.id, "package", package_path, "Writer Package")
    if state_path:
        artifacts["state"] = _file_info(job.id, "state", state_path, "运行状态 JSON")
    if job.log_path:
        artifacts["log"] = _file_info(job.id, "log", job.log_path, "运行日志")

    jobs.update(job, artifacts=artifacts, delivery_status=delivery)


def _append_log(job: ReportJob, text: str) -> None:
    line = text.rstrip("\r\n")
    if not line:
        return
    with open(job.log_path, "a", encoding="utf-8", errors="replace") as handle:
        handle.write(line + "\n")
    with jobs._lock:
        job.log_tail.append(line)


def _classify_finished_job(job: ReportJob) -> str:
    refresh_artifacts(job)
    delivery = job.delivery_status
    if job.artifacts.get("final") and not delivery.get("review_required") and job.returncode == 0:
        return "completed"
    if job.artifacts.get("review_draft") or job.artifacts.get("diagnostic") or job.artifacts.get("fallback"):
        return "needs_review"
    if job.returncode == 0 and job.artifacts.get("writer"):
        return "completed"
    return "failed"


def _run_job(job: ReportJob) -> None:
    jobs.update(job, status="running", started_at=time.time())
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    python_exe = PYTHON_EXE if PYTHON_EXE.exists() else Path(sys.executable)
    command = [
        str(python_exe),
        str(RUN_FULL_REPORT),
        "--query",
        job.query,
        "--route",
        job.route,
        "--output-dir",
        str(job.output_dir),
        "--no-interactive-input",
        "--no-progress-bar",
    ]
    if job.llm_profile:
        command.extend(["--llm-profile", job.llm_profile])
    if job.skip_reformatter:
        command.append("--skip-reformatter")
    jobs.update(job, command=command)
    _append_log(job, " ".join(command))

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "REPORT_NO_INTERACTIVE_INPUT": "1",
            "REPORT_SELECT_LLM_PROFILE": "0",
        }
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=str(PIPELINE_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        jobs.update(job, process=process)
        assert process.stdout is not None
        for line in process.stdout:
            _append_log(job, line)
        returncode = process.wait()
        jobs.update(job, returncode=returncode, finished_at=time.time(), process=None)
        jobs.update(job, status=_classify_finished_job(job))
    except Exception as exc:
        jobs.update(job, status="failed", error=str(exc), finished_at=time.time(), process=None)
        _append_log(job, f"[WEB_ADAPTER_ERROR] {exc}")
        refresh_artifacts(job)


def _job_payload(job: ReportJob) -> Dict[str, Any]:
    refresh_artifacts(job)
    elapsed = None
    if job.started_at:
        elapsed = (job.finished_at or time.time()) - job.started_at
    return {
        "id": job.id,
        "main_title": job.main_title,
        "research_direction": job.research_direction,
        "query": job.query,
        "route": job.route,
        "llm_profile": job.llm_profile,
        "skip_reformatter": job.skip_reformatter,
        "status": job.status,
        "created_at": _now_iso(job.created_at),
        "started_at": _now_iso(job.started_at),
        "finished_at": _now_iso(job.finished_at),
        "elapsed_seconds": round(elapsed, 1) if elapsed is not None else 0,
        "returncode": job.returncode,
        "error": job.error,
        "output_dir": str(job.output_dir),
        "artifacts": job.artifacts,
        "delivery_status": job.delivery_status,
        "log_tail": list(job.log_tail)[-120:],
    }


def _resolve_download(job: ReportJob, kind: str) -> Path:
    refresh_artifacts(job)
    info = job.artifacts.get(kind)
    if not info:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = Path(str(info.get("path") or "")).resolve()
    try:
        path.relative_to(job.output_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="artifact path is outside this job") from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact file missing")
    return path


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "pipeline_root": str(PIPELINE_ROOT),
        "run_full_report": str(RUN_FULL_REPORT),
        "python": str(PYTHON_EXE if PYTHON_EXE.exists() else Path(sys.executable)),
    }


@app.get("/api/reports")
def list_reports() -> Dict[str, Any]:
    return {"items": [_job_payload(job) for job in jobs.list_recent()]}


@app.post("/api/reports")
def create_report(request: ReportRequest) -> Dict[str, Any]:
    if not RUN_FULL_REPORT.exists():
        raise HTTPException(status_code=500, detail=f"run_full_report.py not found: {RUN_FULL_REPORT}")
    job_id = uuid.uuid4().hex[:12]
    output_dir = JOB_OUTPUT_ROOT / job_id
    main_title = " ".join(request.main_title.split())
    research_direction = request.research_direction.strip()
    job = ReportJob(
        id=job_id,
        main_title=main_title,
        research_direction=research_direction,
        query=compose_report_query(main_title, research_direction),
        route=DEFAULT_REPORT_ROUTE,
        llm_profile=request.llm_profile.strip(),
        skip_reformatter=request.skip_reformatter,
        output_dir=output_dir,
        log_path=output_dir / "run.log",
    )
    jobs.add(job)
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return _job_payload(job)


@app.get("/api/reports/{job_id}")
def get_report(job_id: str) -> Dict[str, Any]:
    return _job_payload(jobs.get(job_id))


@app.get("/api/reports/{job_id}/files/{kind}")
def download_artifact(job_id: str, kind: str) -> FileResponse:
    path = _resolve_download(jobs.get(job_id), kind)
    return FileResponse(path, filename=path.name)


@app.get("/api/reports/{job_id}/content/{kind}")
def read_artifact(job_id: str, kind: str) -> PlainTextResponse:
    path = _resolve_download(jobs.get(job_id), kind)
    if path.suffix.lower() not in {".md", ".txt", ".log"}:
        raise HTTPException(status_code=400, detail="artifact is not text-previewable")
    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Missing uvicorn. Install it with: python -m pip install uvicorn") from exc
    host = os.getenv("REPORT_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("REPORT_WEB_PORT", "7888"))
    uvicorn.run("report_web_app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
