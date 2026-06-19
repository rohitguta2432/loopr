"""The optimization loop: run -> score -> reflect -> rewrite, keep the best.

This is the deterministic heart of Loopr. The LLM only (a) runs the candidate
prompt on each case and (b) phrases the reflection. Scoring, best-tracking, and
the stop decision are pure Python, so a run is reproducible for a fixed model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from loopr.llm import LLMClient
from loopr.reflect import propose
from loopr.scorers import get_scorer
from loopr.task import Case, Task


@dataclass
class CaseResult:
    """Score for one case under one candidate prompt."""

    input: str
    expected: str
    output: str
    score: float


@dataclass
class Iteration:
    """One pass of the loop over the active cases."""

    n: int
    prompt: str
    mean_score: float
    case_results: list[CaseResult]
    is_best: bool
    reflection: str = ""
    proposed_prompt: str = ""

    def passed(self) -> int:
        return sum(1 for r in self.case_results if r.score >= 1.0)


@dataclass
class OptimizeResult:
    """The outcome of an optimization run."""

    task_name: str
    seed_prompt: str
    best_prompt: str
    seed_score: float
    best_score: float
    best_iteration: int
    stop_reason: str
    iterations: list[Iteration] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        return self.best_score - self.seed_score

    def to_dict(self) -> dict:
        return {
            "task": self.task_name,
            "seed_score": round(self.seed_score, 4),
            "best_score": round(self.best_score, 4),
            "improvement": round(self.improvement, 4),
            "best_iteration": self.best_iteration,
            "stop_reason": self.stop_reason,
            "best_prompt": self.best_prompt,
            "seed_prompt": self.seed_prompt,
            "iterations": [
                {
                    "n": it.n,
                    "mean_score": round(it.mean_score, 4),
                    "passed": it.passed(),
                    "total": len(it.case_results),
                    "is_best": it.is_best,
                    "prompt": it.prompt,
                    "proposed_prompt": it.proposed_prompt,
                    "reflection": it.reflection,
                    "cases": [
                        {
                            "input": r.input,
                            "expected": r.expected,
                            "output": r.output,
                            "score": round(r.score, 4),
                        }
                        for r in it.case_results
                    ],
                }
                for it in self.iterations
            ],
        }


def evaluate(
    client: LLMClient,
    task: Task,
    prompt: str,
    cases: list[Case],
) -> tuple[float, list[CaseResult]]:
    """Run ``prompt`` over ``cases`` and score each output. Returns (mean, results)."""
    scorer = get_scorer(task.scorer)
    judge = client.generate if task.scorer == "llm_judge" else None
    results: list[CaseResult] = []
    for case in cases:
        rendered = task.render(prompt, case)
        output = client.generate(rendered, task.system)
        score = scorer(output, case.expected, task.scorer_config, judge)
        results.append(CaseResult(case.input, case.expected, output, score))
    mean = sum(r.score for r in results) / len(results) if results else 0.0
    return mean, results


def optimize(
    task: Task,
    client: Optional[LLMClient] = None,
    on_iteration: Optional[Callable[[Iteration], None]] = None,
) -> OptimizeResult:
    """Evolve ``task``'s seed prompt until it converges, plateaus, or runs out of budget.

    ``on_iteration`` is called after each iteration is scored (handy for live progress).
    """
    if client is None:
        client = LLMClient(temperature=task.config.temperature)

    cases = task.active_cases()
    current = task.seed_prompt
    best_prompt = current
    best_score = -1.0
    best_iteration = 0
    seed_score = 0.0
    no_improve = 0
    iterations: list[Iteration] = []
    stop_reason = "budget"

    budget = max(1, task.config.budget)
    for n in range(budget):
        mean, results = evaluate(client, task, current, cases)
        if n == 0:
            seed_score = mean

        is_best = mean > best_score
        if is_best:
            best_prompt, best_score, best_iteration = current, mean, n
            no_improve = 0
        else:
            no_improve += 1

        iteration = Iteration(
            n=n, prompt=current, mean_score=mean, case_results=results, is_best=is_best
        )
        iterations.append(iteration)
        if on_iteration is not None:
            on_iteration(iteration)

        # --- deterministic stop checks ---
        if mean >= task.config.target_score:
            stop_reason = "converged"
            break
        if no_improve >= task.config.patience:
            stop_reason = "plateau"
            break
        if n == budget - 1:
            stop_reason = "budget"
            break

        # --- reflect on failures and propose the next candidate ---
        failures = [r for r in results if r.score < 1.0]
        new_prompt, reflection = propose(client, task, current, failures)
        iteration.reflection = reflection
        iteration.proposed_prompt = new_prompt
        current = new_prompt

    return OptimizeResult(
        task_name=task.name,
        seed_prompt=task.seed_prompt,
        best_prompt=best_prompt,
        seed_score=seed_score,
        best_score=best_score,
        best_iteration=best_iteration,
        stop_reason=stop_reason,
        iterations=iterations,
    )
