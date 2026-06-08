"""P2: a hard global cap on total search tasks. Per-lane caps multiply and cannot
bound the total (the explosion that drags a run into timeout fail-open). The cap
selects round-robin across lanes so coverage is preserved, not flat-truncated."""
from __future__ import annotations

from rag_pipeline.agents.brain_agent import _apply_global_task_budget


def test_global_budget_round_robin_preserves_lane_coverage():
    tasks = (
        [{"scheduled_lane": "market", "task_id": f"m{i}"} for i in range(10)]
        + [{"scheduled_lane": "policy", "task_id": f"p{i}"} for i in range(10)]
        + [{"scheduled_lane": "tech", "task_id": f"t{i}"} for i in range(10)]
    )
    kept, overflow = _apply_global_task_budget(tasks, 6)

    assert len(kept) == 6
    assert len(overflow) == 24
    # round-robin => every lane represented (2 each), not 6 from a single lane
    counts: dict[str, int] = {}
    for task in kept:
        counts[task["scheduled_lane"]] = counts.get(task["scheduled_lane"], 0) + 1
    assert counts == {"market": 2, "policy": 2, "tech": 2}
    assert all(item["drop_reason"] == "global_task_budget" for item in overflow)


def test_global_budget_under_cap_keeps_all():
    tasks = [{"scheduled_lane": "m", "task_id": "1"}, {"scheduled_lane": "p", "task_id": "2"}]
    kept, overflow = _apply_global_task_budget(tasks, 10)
    assert len(kept) == 2
    assert overflow == []


def test_global_budget_uneven_lanes_fills_to_budget():
    # one big lane + two tiny lanes; round-robin still fills exactly to budget
    tasks = (
        [{"scheduled_lane": "big", "task_id": f"b{i}"} for i in range(20)]
        + [{"scheduled_lane": "s1", "task_id": "s1a"}]
        + [{"scheduled_lane": "s2", "task_id": "s2a"}]
    )
    kept, overflow = _apply_global_task_budget(tasks, 8)
    assert len(kept) == 8
    assert len(overflow) == 14
    kept_lanes = {task["scheduled_lane"] for task in kept}
    # the two singleton lanes survive (not starved by the big lane)
    assert {"s1", "s2"} <= kept_lanes
