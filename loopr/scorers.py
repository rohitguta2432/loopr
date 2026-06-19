"""Deterministic scorers. Each returns a float in [0.0, 1.0].

A scorer is the benchmark harness half of the loop - it decides, without an LLM
in the way (except ``llm_judge``, which is explicit), whether an output is right.
Keeping scoring deterministic is what makes Loopr's convergence reproducible.

Signature: ``fn(output, expected, config=None, judge=None) -> float``
  - ``output``   : the model's raw completion
  - ``expected`` : the gold answer (string) from the eval case
  - ``config``   : per-task scorer options (dict)
  - ``judge``    : a ``callable(prompt) -> str`` used only by ``llm_judge``
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def extract_json(text: str) -> str:
    """Return the first balanced ``{...}`` or ``[...]`` block found in ``text``.

    Models love to wrap JSON in prose or code fences; this digs it out so the
    JSON scorers do not depend on perfectly clean output.
    """
    if not text:
        return ""
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return ""


def exact(output: str, expected: str, config: Optional[dict] = None, judge=None) -> float:
    """1.0 iff the normalized output equals the normalized expected answer."""
    return 1.0 if _norm(output) == _norm(expected) else 0.0


def contains(output: str, expected: str, config: Optional[dict] = None, judge=None) -> float:
    """1.0 iff the expected answer appears anywhere in the output (case-insensitive)."""
    return 1.0 if _norm(expected) and _norm(expected) in _norm(output) else 0.0


def regex(output: str, expected: str, config: Optional[dict] = None, judge=None) -> float:
    """1.0 iff a pattern matches the output. Pattern from config['pattern'] or expected."""
    config = config or {}
    pattern = config.get("pattern", expected)
    flags = re.IGNORECASE if config.get("ignorecase", True) else 0
    try:
        return 1.0 if re.search(pattern, output or "", flags) else 0.0
    except re.error:
        return 0.0


def json_field(output: str, expected: str, config: Optional[dict] = None, judge=None) -> float:
    """1.0 iff a dotted JSON field in the output equals the expected value.

    config['field'] selects the field, e.g. "sentiment" or "result.label".
    """
    config = config or {}
    field = config.get("field")
    if not field:
        return 0.0
    blob = extract_json(output)
    if not blob:
        return 0.0
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        return 0.0
    value = obj
    for part in str(field).split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return 0.0
    return 1.0 if _norm(str(value)) == _norm(expected) else 0.0


_JUDGE_SYSTEM = (
    "You are a strict grader. Decide whether the CANDIDATE answer is correct "
    "given the EXPECTED answer. Reply with exactly one word: YES or NO."
)


def llm_judge(output: str, expected: str, config: Optional[dict] = None, judge: Optional[Callable[[str, str], str]] = None) -> float:
    """1.0 iff an LLM judge rules the output correct vs. the expected answer.

    Requires a ``judge`` callable (``client.generate``). ``config['rubric']`` can
    supply task-specific grading criteria.
    """
    if judge is None:
        raise ValueError("llm_judge scorer requires a `judge` callable")
    config = config or {}
    rubric = config.get("rubric", "")
    prompt = (
        (f"GRADING RUBRIC:\n{rubric}\n\n" if rubric else "")
        + f"EXPECTED:\n{expected}\n\nCANDIDATE:\n{output}\n\n"
        "Is the CANDIDATE correct? Reply YES or NO."
    )
    verdict = judge(prompt, _JUDGE_SYSTEM)
    return 1.0 if "yes" in _norm(verdict)[:5] else 0.0


SCORERS = {
    "exact": exact,
    "contains": contains,
    "regex": regex,
    "json_field": json_field,
    "llm_judge": llm_judge,
}


def get_scorer(name: str):
    """Look up a scorer by name, or raise a helpful error listing valid names."""
    try:
        return SCORERS[name]
    except KeyError:
        valid = ", ".join(sorted(SCORERS))
        raise ValueError(f"unknown scorer {name!r}; valid scorers: {valid}") from None
