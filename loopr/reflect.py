"""The reflection step - GEPA-lite prompt evolution.

Given the current prompt and the cases it got wrong, ask the model to (1) diagnose
why it failed in plain language and (2) propose a rewritten prompt. The new prompt
is returned inside ``<prompt>...</prompt>`` markers so it can be parsed reliably.
If parsing fails, the current prompt is returned unchanged - the loop then treats
that as a no-improvement step, so a bad reflection can never silently corrupt state.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle
    from loopr.loop import CaseResult
    from loopr.task import Task

REFLECT_SYSTEM = (
    "You are a prompt-optimization engine. You improve an instruction prompt so a "
    "language model gets more eval cases right. You reason about WHY the current "
    "prompt failed, then rewrite it. Keep the {input} placeholder. Do not solve the "
    "individual cases or hard-code their answers - improve the general instruction."
)

_PROMPT_RE = re.compile(r"<prompt>\s*(.*?)\s*</prompt>", re.DOTALL | re.IGNORECASE)


def _format_failures(task: "Task", failures: list["CaseResult"], limit: int = 8) -> str:
    lines = []
    for i, r in enumerate(failures[:limit], 1):
        got = (r.output or "").strip().replace("\n", " ")
        if len(got) > 200:
            got = got[:200] + "..."
        lines.append(
            f"{i}. INPUT: {r.input}\n   EXPECTED: {r.expected}\n   GOT: {got}"
        )
    extra = len(failures) - limit
    if extra > 0:
        lines.append(f"...and {extra} more failing case(s) in the same vein.")
    return "\n".join(lines)


def build_reflection_prompt(task: "Task", current_prompt: str, failures: list["CaseResult"]) -> str:
    """Assemble the reflection prompt sent to the reflector model."""
    goal = task.description or task.name
    return (
        f"TASK GOAL: {goal}\n\n"
        f"CURRENT PROMPT:\n\"\"\"\n{current_prompt}\n\"\"\"\n\n"
        f"THIS PROMPT FAILED ON THESE CASES:\n{_format_failures(task, failures)}\n\n"
        "Step 1 - Diagnose: in 1-3 sentences, say what about the prompt likely caused "
        "these failures (ambiguity, missing format constraint, wrong label set, etc.).\n"
        "Step 2 - Rewrite: produce an improved prompt that fixes the diagnosis. It must "
        "keep the {input} placeholder and must generalize - never name or answer the "
        "specific cases above.\n\n"
        "Return ONLY the rewritten prompt wrapped exactly like this:\n"
        "<prompt>\n...your improved prompt here...\n</prompt>"
    )


def parse_proposed_prompt(text: str) -> str:
    """Pull the rewritten prompt out of ``<prompt>...</prompt>``; '' if absent."""
    match = _PROMPT_RE.search(text or "")
    return match.group(1).strip() if match else ""


def propose(client, task: "Task", current_prompt: str, failures: list["CaseResult"]) -> str:
    """Ask the model for an improved prompt. Falls back to the current prompt.

    Returns ``(new_prompt, raw_reflection)``.
    """
    reflection_prompt = build_reflection_prompt(task, current_prompt, failures)
    raw = client.generate(reflection_prompt, REFLECT_SYSTEM)
    proposed = parse_proposed_prompt(raw)
    if not proposed or "{input}" not in proposed:
        # Could not parse a usable prompt - keep the current one (no-op step).
        return current_prompt, raw
    return proposed, raw
