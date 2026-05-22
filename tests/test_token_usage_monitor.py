from __future__ import annotations

import importlib
import json


def test_normalize_usage_supports_chat_and_responses():
    from rag_pipeline.telemetry.token_usage import normalize_usage

    chat = normalize_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    responses = normalize_usage({"input_tokens": 7, "output_tokens": 3})

    assert chat["input_tokens"] == 10
    assert chat["output_tokens"] == 5
    assert chat["total_tokens"] == 15
    assert responses["input_tokens"] == 7
    assert responses["output_tokens"] == 3
    assert responses["total_tokens"] == 10


def test_record_llm_usage_aggregates_and_prints(monkeypatch, capsys):
    monkeypatch.setenv("RAG_TOKEN_MONITOR_ENABLED", "true")
    monkeypatch.setenv("RAG_TOKEN_MONITOR_PRINT", "true")
    monkeypatch.setenv("RAG_TOKEN_MONITOR_PRINT_EACH_CALL", "true")
    monkeypatch.setenv("RAG_TOKEN_MONITOR_JSONL", "false")

    import rag_pipeline.telemetry.token_usage as token_usage

    token_usage = importlib.reload(token_usage)
    event = token_usage.record_llm_usage(
        usage={"input_tokens": 12, "output_tokens": 8},
        provider="openai_compatible",
        model="gpt-5.5",
        task="qa",
        profile="gpt-5.5",
        api="openai_responses_json",
        elapsed_ms=123,
    )

    captured = capsys.readouterr()
    summary = token_usage.token_usage_summary()
    assert event is not None
    assert "[TOKEN] task=qa model=gpt-5.5" in captured.err
    assert summary["call_count"] == 1
    assert summary["total_tokens"] == 20
    assert summary["by_model"]["gpt-5.5"]["total_tokens"] == 20
    assert summary["by_task"]["qa"]["total_tokens"] == 20


def test_memory_responses_adapter_records_usage(monkeypatch):
    monkeypatch.setenv("RAG_TOKEN_MONITOR_ENABLED", "true")
    monkeypatch.setenv("RAG_TOKEN_MONITOR_PRINT", "false")
    monkeypatch.setenv("RAG_TOKEN_MONITOR_JSONL", "false")

    import rag_pipeline.telemetry.token_usage as token_usage
    import rag_pipeline.search.memory as memory

    token_usage = importlib.reload(token_usage)
    memory = importlib.reload(memory)

    def fake_post(**kwargs):
        return {
            "output_text": json.dumps({"status": "pass"}),
            "usage": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
        }

    monkeypatch.setattr(memory, "_post_llm_json", fake_post)
    response = memory.call_openai_compatible_json(
        config={
            "provider": "openai_compatible",
            "url": "https://api.openai.com/v1/responses",
            "api_key": "test-key",
            "model": "gpt-5.5",
            "task_name": "final_audit",
            "profile": "gpt-5.5",
            "reasoning_effort": "high",
        },
        system_prompt="Return JSON.",
        user_payload={"report": "x"},
    )

    summary = token_usage.token_usage_summary()
    assert response["payload"]["status"] == "pass"
    assert summary["call_count"] == 1
    assert summary["by_task"]["final_audit"]["total_tokens"] == 125
    assert summary["by_profile"]["gpt-5.5"]["total_tokens"] == 125


def test_token_monitor_disabled_records_nothing(monkeypatch):
    monkeypatch.setenv("RAG_TOKEN_MONITOR_ENABLED", "false")

    import rag_pipeline.telemetry.token_usage as token_usage

    token_usage = importlib.reload(token_usage)
    event = token_usage.record_llm_usage(
        usage={"total_tokens": 10},
        model="qwen3.6-plus",
        task="reformatter",
    )

    assert event is None
    assert token_usage.token_usage_summary()["call_count"] == 0

