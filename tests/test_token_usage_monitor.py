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


def test_llm_text_preflight_blocks_oversized_input_before_http(monkeypatch):
    monkeypatch.setenv("RAG_LLM_CONTEXT_BUDGET_ENABLED", "true")
    monkeypatch.setenv("RAG_LLM_MAX_INPUT_TOKENS", "8")

    import rag_pipeline.search.memory as memory
    from rag_pipeline.telemetry.context_budget import ContextBudgetExceededError

    memory = importlib.reload(memory)

    def fail_post(**kwargs):
        raise AssertionError("HTTP should not be called when the input budget is exceeded")

    monkeypatch.setattr(memory, "_post_llm_json", fail_post)

    try:
        memory.call_openai_compatible_text(
            config={
                "provider": "openai_compatible",
                "url": "https://api.example.com/v1/chat/completions",
                "api_key": "test-key",
                "model": "test-model",
                "task_name": "oversized_task",
            },
            system_prompt="system",
            user_content="x" * 200,
            max_tokens=10,
        )
    except ContextBudgetExceededError as exc:
        diagnostic = exc.diagnostic
    else:
        raise AssertionError("Expected the input budget guard to block the call")

    assert diagnostic["type"] == "llm_context_budget"
    assert diagnostic["task"] == "oversized_task"
    assert diagnostic["estimated_input_tokens"] > diagnostic["max_input_tokens"]


def test_llm_json_preflight_blocks_by_char_budget_before_http(monkeypatch):
    monkeypatch.setenv("RAG_LLM_CONTEXT_BUDGET_ENABLED", "true")
    monkeypatch.setenv("RAG_LLM_MAX_INPUT_CHARS", "40")

    import rag_pipeline.search.memory as memory
    from rag_pipeline.telemetry.context_budget import ContextBudgetExceededError

    memory = importlib.reload(memory)

    def fail_post(**kwargs):
        raise AssertionError("HTTP should not be called when the input char budget is exceeded")

    monkeypatch.setattr(memory, "_post_llm_json", fail_post)

    try:
        memory.call_openai_compatible_json(
            config={
                "provider": "openai_compatible",
                "url": "https://api.example.com/v1/chat/completions",
                "api_key": "test-key",
                "model": "test-model",
                "task_name": "coverage_eval",
            },
            system_prompt="json",
            user_payload={"items": ["x" * 80]},
        )
    except ContextBudgetExceededError as exc:
        diagnostic = exc.diagnostic
    else:
        raise AssertionError("Expected the input budget guard to block the call")

    assert diagnostic["type"] == "llm_context_budget"
    assert diagnostic["task"] == "coverage_eval"
    assert diagnostic["input_chars"] > diagnostic["max_input_chars"]


def test_context_budget_monitor_records_peak_and_blocked_count(monkeypatch):
    monkeypatch.setenv("RAG_TOKEN_MONITOR_ENABLED", "true")
    monkeypatch.setenv("RAG_TOKEN_MONITOR_PRINT", "false")
    monkeypatch.setenv("RAG_TOKEN_MONITOR_PRINT_CONTEXT_BUDGET", "false")
    monkeypatch.setenv("RAG_TOKEN_MONITOR_JSONL", "false")
    monkeypatch.setenv("RAG_LLM_CONTEXT_BUDGET_ENABLED", "true")

    import rag_pipeline.telemetry.token_usage as token_usage
    import rag_pipeline.telemetry.context_budget as context_budget

    token_usage = importlib.reload(token_usage)
    context_budget = importlib.reload(context_budget)

    normalized_config = {
        "provider": "openai_compatible",
        "model": "test-model",
        "task_name": "budget_task",
        "profile": "test",
    }
    context_budget.assert_llm_input_budget(
        normalized_config=normalized_config,
        system_prompt="system",
        user_content="short",
        api="openai_compatible_chat_text",
        max_output_tokens=16,
    )
    monkeypatch.setenv("RAG_LLM_MAX_INPUT_TOKENS", "1")
    try:
        context_budget.assert_llm_input_budget(
            normalized_config=normalized_config,
            system_prompt="system",
            user_content="x" * 100,
            api="openai_compatible_chat_text",
            max_output_tokens=16,
        )
    except context_budget.ContextBudgetExceededError:
        pass
    else:
        raise AssertionError("Expected the context budget monitor test to block the oversized call")

    budget_summary = token_usage.token_usage_summary()["context_budget"]
    task_summary = budget_summary["by_task"]["budget_task"]
    assert budget_summary["event_count"] == 2
    assert budget_summary["blocked_count"] == 1
    assert task_summary["event_count"] == 2
    assert task_summary["blocked_count"] == 1
    assert task_summary["peak_estimated_input_tokens"] > 1
