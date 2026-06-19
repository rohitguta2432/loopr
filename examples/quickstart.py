"""Minimal Loopr example - optimize a prompt from Python.

Run:  python examples/quickstart.py
Needs a local Ollama (ollama serve) or OPENAI_API_KEY / ANTHROPIC_API_KEY.
"""

from loopr import Task, optimize

task = Task.from_dict(
    {
        "name": "yesno",
        "description": "Answer the question with exactly 'yes' or 'no', lowercase.",
        "seed_prompt": "Question: {input}",  # under-specified on purpose
        "scorer": "exact",
        "config": {"budget": 5, "patience": 2},
        "cases": [
            {"input": "Is the sky blue on a clear day?", "expected": "yes"},
            {"input": "Is 7 an even number?", "expected": "no"},
            {"input": "Do fish breathe air with lungs?", "expected": "no"},
            {"input": "Is water wet?", "expected": "yes"},
        ],
    }
)

result = optimize(task, on_iteration=lambda it: print(f"  iter {it.n}: {it.mean_score:.0%}"))

print(f"\nstop: {result.stop_reason} | seed {result.seed_score:.0%} -> best {result.best_score:.0%}")
print("\nbest prompt:\n" + result.best_prompt)
