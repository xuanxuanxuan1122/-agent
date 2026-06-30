from rag_pipeline.agents.layout_compiler import (
    _detemplate_chapter_title,
    validate_dynamic_blueprint,
)


def test_detemplate_chapter_title_removes_banned_template_words():
    raw = "中国具身智能机器人是否存在真实需求和可验证市场空间，而不是概念热度"
    cleaned = _detemplate_chapter_title(raw)
    # The "……，而不是X" contrast tail is dropped and the fixed template words are rephrased.
    assert "真实需求" not in cleaned
    assert "概念热度" not in cleaned
    assert "而不是" not in cleaned
    assert cleaned.strip()


def test_detemplate_keeps_normal_titles_unchanged():
    title = "竞争格局与关键变量"
    assert _detemplate_chapter_title(title) == title


def test_validate_blueprint_passes_after_detemplating():
    raw_title = "中国具身智能机器人是否存在真实需求和可验证市场空间，而不是概念热度"

    # The raw default-template title is a blocking layout issue (template_like_title).
    raw_result = validate_dynamic_blueprint([{"chapter_title": raw_title}])
    assert raw_result["blocking_count"] >= 1
    assert raw_result["passed"] is False

    # After de-templating, the same chapter is publishable-clean (no blocking issue).
    clean_result = validate_dynamic_blueprint(
        [{"chapter_title": _detemplate_chapter_title(raw_title)}]
    )
    assert clean_result["blocking_count"] == 0
    assert clean_result["passed"] is True
