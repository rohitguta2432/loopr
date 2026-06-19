import json

import pytest

from loopr.task import Case, Task


def _minimal_dict():
    return {
        "name": "t",
        "seed_prompt": "Answer: {input}",
        "scorer": "exact",
        "cases": [{"input": "a", "expected": "a"}, {"input": "b", "expected": "b"}],
    }


def test_from_dict_builds_task():
    task = Task.from_dict(_minimal_dict())
    assert task.name == "t"
    assert len(task.cases) == 2
    assert task.cases[0] == Case(input="a", expected="a")
    assert task.config.budget == 6  # default


def test_from_dict_requires_seed_prompt_and_cases():
    with pytest.raises(ValueError):
        Task.from_dict({"name": "x", "cases": [{"input": "a"}]})
    with pytest.raises(ValueError):
        Task.from_dict({"name": "x", "seed_prompt": "p", "cases": []})


def test_render_replaces_placeholder():
    task = Task.from_dict(_minimal_dict())
    assert task.render("Answer: {input}", Case("hello")) == "Answer: hello"


def test_render_appends_when_no_placeholder():
    task = Task.from_dict(_minimal_dict())
    assert task.render("Classify it", Case("hello")) == "Classify it\n\nhello"


def test_active_cases_honors_minibatch():
    data = _minimal_dict()
    data["config"] = {"minibatch": 1}
    task = Task.from_dict(data)
    assert len(task.active_cases()) == 1


def test_from_file_reads_json(tmp_path):
    p = tmp_path / "task.json"
    p.write_text(json.dumps(_minimal_dict()), encoding="utf-8")
    task = Task.from_file(str(p))
    assert task.name == "t"


def test_bundled_yaml_tasks_load():
    import os

    here = os.path.dirname(os.path.dirname(__file__))
    for name in ("sentiment.yaml", "extract_json.yaml"):
        task = Task.from_file(os.path.join(here, "loopr", "tasks", name))
        assert task.cases
        assert "{input}" in task.seed_prompt
