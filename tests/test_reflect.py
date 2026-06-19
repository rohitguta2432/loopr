from loopr.loop import CaseResult
from loopr.reflect import build_reflection_prompt, parse_proposed_prompt, propose
from loopr.task import Task


def _task():
    return Task.from_dict(
        {
            "name": "t",
            "seed_prompt": "Answer: {input}",
            "scorer": "exact",
            "cases": [{"input": "a", "expected": "a"}],
        }
    )


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, prompt, system=""):
        self.calls.append((prompt, system))
        return self.reply


def test_parse_extracts_prompt_block():
    text = "diagnosis...\n<prompt>\nBetter prompt with {input}\n</prompt>\ntrailing"
    assert parse_proposed_prompt(text) == "Better prompt with {input}"


def test_parse_returns_empty_when_no_block():
    assert parse_proposed_prompt("no markers here") == ""


def test_build_reflection_prompt_includes_failures():
    failures = [CaseResult("in1", "want", "got-wrong", 0.0)]
    text = build_reflection_prompt(_task(), "Answer: {input}", failures)
    assert "in1" in text and "want" in text and "got-wrong" in text
    assert "{input}" in text


def test_propose_returns_parsed_prompt():
    client = FakeClient("<prompt>Improved: {input}</prompt>")
    new_prompt, raw = propose(client, _task(), "Answer: {input}", [])
    assert new_prompt == "Improved: {input}"
    assert "Improved" in raw


def test_propose_falls_back_when_unparseable():
    client = FakeClient("I could not produce a prompt")
    current = "Answer: {input}"
    new_prompt, _ = propose(client, _task(), current, [])
    assert new_prompt == current  # no-op, never corrupts state


def test_propose_falls_back_when_placeholder_dropped():
    client = FakeClient("<prompt>This prompt forgot the placeholder</prompt>")
    current = "Answer: {input}"
    new_prompt, _ = propose(client, _task(), current, [])
    assert new_prompt == current
