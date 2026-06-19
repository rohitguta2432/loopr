from loopr.scorers import (
    contains,
    exact,
    extract_json,
    get_scorer,
    json_field,
    llm_judge,
    regex,
)


def test_exact_is_case_and_space_insensitive():
    assert exact("Positive", "positive") == 1.0
    assert exact("  positive  ", "positive") == 1.0
    assert exact("positively", "positive") == 0.0


def test_contains_matches_substring():
    assert contains("The sentiment is positive.", "positive") == 1.0
    assert contains("nope", "positive") == 0.0
    assert contains("anything", "") == 0.0  # empty expected never matches


def test_regex_uses_pattern_from_config_or_expected():
    assert regex("order #12345", r"#\d+") == 1.0
    assert regex("order abc", r"#\d+") == 0.0
    assert regex("LABEL: yes", "", {"pattern": r"label:\s*yes"}) == 1.0


def test_extract_json_digs_out_a_block_from_prose():
    text = 'Sure! Here you go: {"role": "engineer"} hope that helps'
    assert extract_json(text) == '{"role": "engineer"}'
    assert extract_json("no json here") == ""


def test_extract_json_handles_nested_braces():
    text = 'result: {"a": {"b": 1}} done'
    assert extract_json(text) == '{"a": {"b": 1}}'


def test_json_field_reads_dotted_path():
    out = 'here: {"result": {"label": "Positive"}}'
    assert json_field(out, "positive", {"field": "result.label"}) == 1.0
    assert json_field(out, "negative", {"field": "result.label"}) == 0.0
    assert json_field("not json", "positive", {"field": "result.label"}) == 0.0
    assert json_field('{"x":1}', "positive", {}) == 0.0  # no field configured


def test_llm_judge_reads_yes_no():
    yes = lambda prompt, system: "YES, looks correct"
    no = lambda prompt, system: "No - wrong label"
    assert llm_judge("foo", "bar", {}, judge=yes) == 1.0
    assert llm_judge("foo", "bar", {}, judge=no) == 0.0


def test_get_scorer_lookup_and_error():
    assert get_scorer("exact") is exact
    try:
        get_scorer("does-not-exist")
    except ValueError as exc:
        assert "unknown scorer" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown scorer")
