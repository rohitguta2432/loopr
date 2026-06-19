"""End-to-end loop tests with injected stub LLMs - no network, fully deterministic."""

from loopr.llm import LLMClient
from loopr.loop import evaluate, optimize
from loopr.reflect import REFLECT_SYSTEM
from loopr.task import Task

REFLECT_MARK = "prompt-optimization engine"  # appears in REFLECT_SYSTEM


def _echo_task(**cfg):
    data = {
        "name": "echo",
        "seed_prompt": "Repeat the word: {input}",  # no FIXED marker -> fails
        "scorer": "exact",
        "cases": [{"input": "alpha", "expected": "alpha"}, {"input": "beta", "expected": "beta"}],
        "config": cfg,
    }
    return Task.from_dict(data)


def make_client(generate):
    return LLMClient(generate=generate)


def test_reflect_system_constant_matches_marker():
    assert REFLECT_MARK in REFLECT_SYSTEM


def test_loop_converges_after_reflection():
    """Seed prompt fails; reflection injects a 'FIXED' prompt that makes the
    model echo the input correctly; the loop should converge."""

    def stub(prompt, system):
        if REFLECT_MARK in system:  # reflection call
            return "<prompt>FIXED echo exactly {input}</prompt>"
        if "FIXED" in prompt:  # good task prompt -> echo the trailing input token
            return prompt.strip().split()[-1]
        return "nope"

    result = optimize(_echo_task(budget=5, patience=2), client=make_client(stub))
    assert result.seed_score == 0.0
    assert result.best_score == 1.0
    assert result.stop_reason == "converged"
    assert "FIXED" in result.best_prompt
    assert len(result.iterations) == 2  # seed fails, rewritten passes


def test_loop_plateaus_when_reflection_never_helps():
    def stub(prompt, system):
        if REFLECT_MARK in system:
            return "no usable prompt block here"  # propose() falls back -> no change
        return "nope"

    result = optimize(_echo_task(budget=10, patience=2), client=make_client(stub))
    assert result.best_score == 0.0
    assert result.stop_reason == "plateau"
    # iter0 (best, reset), iter1 (+1), iter2 (+1 -> hits patience)
    assert len(result.iterations) == 3


def test_loop_stops_at_budget():
    def stub(prompt, system):
        if REFLECT_MARK in system:
            return "still no block"
        return "nope"

    # patience high so plateau never fires; budget is the binding constraint
    result = optimize(_echo_task(budget=3, patience=99), client=make_client(stub))
    assert result.stop_reason == "budget"
    assert len(result.iterations) == 3


def test_loop_respects_target_score():
    # Stub answers "alpha" for everything: case alpha passes, case beta fails -> 0.5
    def stub(prompt, system):
        return "alpha"

    result = optimize(_echo_task(budget=5, target_score=0.5), client=make_client(stub))
    assert result.seed_score == 0.5
    assert result.stop_reason == "converged"
    assert result.best_iteration == 0


def test_evaluate_returns_per_case_results():
    def stub(prompt, system):
        return "alpha"

    task = _echo_task()
    mean, results = evaluate(make_client(stub), task, task.seed_prompt, task.cases)
    assert mean == 0.5
    assert len(results) == 2
    assert results[0].score == 1.0 and results[1].score == 0.0


def test_on_iteration_callback_fires():
    seen = []

    def stub(prompt, system):
        return "alpha"

    optimize(
        _echo_task(budget=2, target_score=0.5),
        client=make_client(stub),
        on_iteration=lambda it: seen.append(it.n),
    )
    assert seen == [0]  # converged immediately at iter 0
