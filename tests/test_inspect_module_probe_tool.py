import json
import subprocess
import sys
from pathlib import Path


def test_inspect_module_probe_reports_root_hygiene_for_offline_probe(tmp_path):
    probe_path = tmp_path / "run.module_probe.jsonl"
    probe_path.write_text(
        json.dumps(
            {
                "schema_version": "module_probe_event_v1",
                "run_id": "run",
                "seq": 1,
                "stage": "evidence_merger",
                "module": "evidence_merger",
                "event_type": "transform_result",
                "status": "warning",
                "input": {"count": 2},
                "output": {"count": 1},
                "drop": {"count": 1, "reason_counts": {}},
                "cache": {"hit_count": 0},
                "diagnostics": {
                    "evidence_root_hygiene": {
                        "dirty_item_count": 1,
                        "reason_counts": {"artifact_like_value": 1},
                    }
                },
                "diagnostic_only": True,
                "must_not_render": True,
                "public_text_allowed": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "tools/inspect_module_probe.py", str(probe_path), "--json"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["runtime"]["root_hygiene"]["status"] == "warning"
    assert payload["runtime"]["root_hygiene"]["earliest_issue_stage"] == "evidence_merger"
    assert payload["runtime"]["root_hygiene"]["reason_counts"]["evidence:artifact_like_value"] == 1
