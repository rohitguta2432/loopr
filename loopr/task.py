"""Task spec: the prompt to optimize, the eval cases, and the loop budget.

A task can be authored in YAML or JSON. Minimal shape::

    name: sentiment
    seed_prompt: |
      Classify the sentiment. Answer in one word.
      Text: {input}
    scorer: contains
    cases:
      - {input: "I love it!", expected: positive}
      - {input: "Worst ever.", expected: negative}

The ``{input}`` placeholder in the prompt is replaced with each case's input.
If the prompt has no ``{input}``, the input is appended on its own line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Case:
    """One eval example: an input and the gold answer it should produce."""

    input: str
    expected: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Loop controls.

    budget   : max iterations (prompt rewrites + the initial eval).
    patience : stop after this many iterations with no score improvement.
    minibatch: evaluate on the first N cases each iteration (None = all).
    target_score: stop early once mean score reaches this (1.0 = perfect).
    temperature : sampling temperature for task runs (0.0 = deterministic).
    """

    budget: int = 6
    patience: int = 2
    minibatch: Optional[int] = None
    target_score: float = 1.0
    temperature: float = 0.0


@dataclass
class Task:
    name: str
    seed_prompt: str
    cases: list[Case]
    description: str = ""
    scorer: str = "contains"
    scorer_config: dict = field(default_factory=dict)
    system: str = ""
    config: TaskConfig = field(default_factory=TaskConfig)

    def render(self, prompt: str, case: Case) -> str:
        """Substitute a case's input into a candidate prompt."""
        if "{input}" in prompt:
            return prompt.replace("{input}", case.input)
        return f"{prompt}\n\n{case.input}"

    def active_cases(self) -> list[Case]:
        """The cases used per iteration, honoring ``config.minibatch``."""
        if self.config.minibatch:
            return self.cases[: self.config.minibatch]
        return self.cases

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Task":
        if "seed_prompt" not in data:
            raise ValueError("task is missing required field 'seed_prompt'")
        raw_cases = data.get("cases") or []
        if not raw_cases:
            raise ValueError("task must define at least one case under 'cases'")
        cases = [
            Case(
                input=str(c["input"]),
                expected=str(c.get("expected", "")),
                meta=dict(c.get("meta", {})),
            )
            for c in raw_cases
        ]
        cfg_data = data.get("config") or {}
        config = TaskConfig(
            budget=int(cfg_data.get("budget", 6)),
            patience=int(cfg_data.get("patience", 2)),
            minibatch=cfg_data.get("minibatch"),
            target_score=float(cfg_data.get("target_score", 1.0)),
            temperature=float(cfg_data.get("temperature", 0.0)),
        )
        return Task(
            name=str(data.get("name", "task")),
            seed_prompt=str(data["seed_prompt"]),
            cases=cases,
            description=str(data.get("description", "")),
            scorer=str(data.get("scorer", "contains")),
            scorer_config=dict(data.get("scorer_config", {})),
            system=str(data.get("system", "")),
            config=config,
        )

    @staticmethod
    def from_file(path: str) -> "Task":
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            import yaml  # local import so JSON-only users need no PyYAML at import time

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return Task.from_dict(data)
