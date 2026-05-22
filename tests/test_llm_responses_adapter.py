from __future__ import annotations

import json

from rag_pipeline.search import memory


def test_gpt55_json_uses_responses_api_with_high_reasoning(monkeypatch):
    captured = {}

    def fake_post_llm_json(*, normalized, url, payload, error_prefix):
        captured["normalized"] = normalized
        captured["url"] = url
        captured["payload"] = payload
        captured["error_prefix"] = error_prefix
        return {"output_text": '{"status":"pass","overall_score":95}', "usage": {"total_tokens": 12}}

    monkeypatch.setattr(memory, "_post_llm_json", fake_post_llm_json)

    result = memory.call_openai_compatible_json(
        config={
            "provider": "openai_compatible",
            "url": "https://api.openai.com/v1/chat/completions",
            "api_key": "sk-test",
            "model": "gpt-5.5",
            "timeout": 240,
            "reasoning_effort": "high",
            "max_output_tokens": 32000,
        },
        system_prompt="Return JSON.",
        user_payload={"task": "audit"},
    )

    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["payload"]["model"] == "gpt-5.5"
    assert captured["payload"]["reasoning"] == {"effort": "high"}
    assert captured["payload"]["max_output_tokens"] == 32000
    assert captured["payload"]["text"]["format"]["type"] == "json_object"
    assert "json" in captured["payload"]["input"].lower()
    assert json.loads(captured["payload"]["input"])["payload"]["task"] == "audit"
    assert "temperature" not in captured["payload"]
    assert "messages" not in captured["payload"]
    assert result["payload"]["status"] == "pass"
    assert result["llm_call"]["model"] == "gpt-5.5"
    assert result["llm_call"]["api"] == "openai_responses_json"
    assert result["llm_call"]["status"] == "success"


def test_deepseek_v4_disable_thinking_uses_thinking_object(monkeypatch):
    captured = {}

    def fake_post_llm_json(*, normalized, url, payload, error_prefix):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": '{"ok":true}'}}], "usage": {}}

    monkeypatch.setattr(memory, "_post_llm_json", fake_post_llm_json)

    memory.call_openai_compatible_json(
        config={
            "provider": "openai_compatible",
            "url": "https://api.deepseek.com/chat/completions",
            "api_key": "ds-test",
            "model": "deepseek-v4-pro",
            "disable_thinking": True,
        },
        system_prompt="Return JSON.",
        user_payload={"task": "test"},
    )

    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in captured["payload"]


def test_gpt55_json_falls_back_to_deepseek_and_circuits(monkeypatch):
    memory._UNAVAILABLE_LLM_KEYS.clear()
    calls = []

    def fake_post_llm_json(*, normalized, url, payload, error_prefix):
        calls.append({"model": normalized["model"], "url": url})
        if normalized["model"] == "gpt-5.5":
            raise RuntimeError("LLM request failed with HTTP 429: insufficient_quota")
        return {"choices": [{"message": {"content": '{"ok":true}'}}], "usage": {"total_tokens": 3}}

    monkeypatch.setattr(memory, "_post_llm_json", fake_post_llm_json)
    config = {
        "provider": "openai_compatible",
        "url": "https://api.openai.com/v1/responses",
        "api_key": "sk-test",
        "model": "gpt-5.5",
        "profile": "gpt-5.5",
        "fallback_config": {
            "provider": "openai_compatible",
            "url": "https://api.deepseek.com/chat/completions",
            "api_key": "ds-test",
            "model": "deepseek-v4-pro",
            "profile": "deepseek-v4-pro",
        },
    }

    first = memory.call_openai_compatible_json(config=config, system_prompt="JSON", user_payload={"task": "x"})
    second = memory.call_openai_compatible_json(config=config, system_prompt="JSON", user_payload={"task": "y"})

    assert first["payload"]["ok"] is True
    assert first["llm_call"]["fallback_used"] is True
    assert first["llm_call"]["primary_model"] == "gpt-5.5"
    assert first["llm_call"]["fallback_model"] == "deepseek-v4-pro"
    assert second["llm_call"]["primary_skipped"] is True
    assert [call["model"] for call in calls] == ["gpt-5.5", "deepseek-v4-pro", "deepseek-v4-pro"]
    memory._UNAVAILABLE_LLM_KEYS.clear()


def test_gpt55_text_falls_back_to_deepseek(monkeypatch):
    memory._UNAVAILABLE_LLM_KEYS.clear()
    calls = []

    def fake_post_llm_json(*, normalized, url, payload, error_prefix):
        calls.append(normalized["model"])
        if normalized["model"] == "gpt-5.5":
            raise RuntimeError("LLM request failed with HTTP 429: insufficient_quota")
        return {"choices": [{"message": {"content": "fallback text"}}], "usage": {"total_tokens": 2}}

    monkeypatch.setattr(memory, "_post_llm_json", fake_post_llm_json)

    result = memory.call_openai_compatible_text(
        config={
            "provider": "openai_compatible",
            "url": "https://api.openai.com/v1/responses",
            "api_key": "sk-test",
            "model": "gpt-5.5",
            "profile": "gpt-5.5",
            "fallback_config": {
                "provider": "openai_compatible",
                "url": "https://api.deepseek.com/chat/completions",
                "api_key": "ds-test",
                "model": "deepseek-v4-pro",
                "profile": "deepseek-v4-pro",
            },
        },
        system_prompt="Text",
        user_content="hello",
    )

    assert result["text"] == "fallback text"
    assert result["llm_call"]["fallback_used"] is True
    assert calls == ["gpt-5.5", "deepseek-v4-pro"]
    memory._UNAVAILABLE_LLM_KEYS.clear()
