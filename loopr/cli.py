"""Command-line interface for Loopr.

    loopr optimize <task.yaml>          evolve the prompt, write the trace + best prompt
    loopr eval <task.yaml> [--prompt-file P]   score the seed (or a given) prompt once
    loopr run <task.yaml> --input "..." [--prompt-file P]   run one input through a prompt
    loopr version
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from loopr import __version__
from loopr.llm import LLMClient, LLMError
from loopr.loop import Iteration, evaluate, optimize
from loopr.task import Case, Task

BAR_WIDTH = 24


def _bar(score: float) -> str:
    filled = int(round(score * BAR_WIDTH))
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def _client(args, temperature: float) -> LLMClient:
    return LLMClient(model=args.model, temperature=temperature)


def _load_prompt(path: str | None) -> str | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def _print_iteration(it: Iteration) -> None:
    marker = " *best*" if it.is_best else ""
    print(
        f"  iter {it.n}: {_bar(it.mean_score)} "
        f"{it.mean_score:5.0%}  ({it.passed()}/{len(it.case_results)} cases){marker}"
    )


def cmd_optimize(args) -> int:
    task = Task.from_file(args.task)
    if args.budget is not None:
        task.config.budget = args.budget
    if args.patience is not None:
        task.config.patience = args.patience

    print(f"\n🔁 loopr · optimizing '{task.name}' "
          f"(scorer={task.scorer}, budget={task.config.budget}, cases={len(task.active_cases())})\n")
    client = _client(args, task.config.temperature)
    try:
        result = optimize(task, client=client, on_iteration=_print_iteration)
    except LLMError as exc:
        print(f"\n✗ LLM backend unavailable: {exc}", file=sys.stderr)
        print("  Start Ollama (ollama serve) or set OPENAI_API_KEY / ANTHROPIC_API_KEY.", file=sys.stderr)
        return 2

    print(
        f"\n✅ stop: {result.stop_reason}  |  "
        f"seed {result.seed_score:.0%} → best {result.best_score:.0%} "
        f"(+{result.improvement:.0%}, iter {result.best_iteration})\n"
    )
    print("── best prompt " + "─" * 40)
    print(result.best_prompt)
    print("─" * 54)

    os.makedirs(args.out, exist_ok=True)
    trace_path = os.path.join(args.out, f"{task.name}.trace.json")
    prompt_path = os.path.join(args.out, f"{task.name}.best_prompt.txt")
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
    with open(prompt_path, "w", encoding="utf-8") as fh:
        fh.write(result.best_prompt + "\n")
    print(f"\n📎 trace → {trace_path}\n📎 best  → {prompt_path}")
    return 0


def cmd_eval(args) -> int:
    task = Task.from_file(args.task)
    prompt = _load_prompt(args.prompt_file) or task.seed_prompt
    client = _client(args, task.config.temperature)
    try:
        mean, results = evaluate(client, task, prompt, task.active_cases())
    except LLMError as exc:
        print(f"✗ LLM backend unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"\n📊 '{task.name}' scored {mean:.0%} ({sum(1 for r in results if r.score >= 1)}/{len(results)})\n")
    for r in results:
        mark = "✓" if r.score >= 1.0 else "✗"
        got = r.output.strip().replace("\n", " ")
        if len(got) > 70:
            got = got[:70] + "..."
        print(f"  {mark} [{r.score:.0%}] expected={r.expected!r}  got={got!r}")
    return 0


def cmd_run(args) -> int:
    task = Task.from_file(args.task)
    prompt = _load_prompt(args.prompt_file) or task.seed_prompt
    client = _client(args, task.config.temperature)
    rendered = task.render(prompt, Case(input=args.input))
    try:
        print(client.generate(rendered, task.system))
    except LLMError as exc:
        print(f"✗ LLM backend unavailable: {exc}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopr", description="Self-improving prompt-optimization loop.")
    parser.add_argument("--model", default=None, help="LLM model (default: $LOOPR_MODEL or qwen2.5:14b)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_opt = sub.add_parser("optimize", help="evolve the prompt until it converges")
    p_opt.add_argument("task", help="path to a task .yaml / .json")
    p_opt.add_argument("--out", default="loopr-out", help="output directory for trace + best prompt")
    p_opt.add_argument("--budget", type=int, default=None, help="override max iterations")
    p_opt.add_argument("--patience", type=int, default=None, help="override plateau patience")
    p_opt.set_defaults(func=cmd_optimize)

    p_eval = sub.add_parser("eval", help="score the seed (or a given) prompt once")
    p_eval.add_argument("task", help="path to a task .yaml / .json")
    p_eval.add_argument("--prompt-file", default=None, help="score this prompt instead of the seed")
    p_eval.set_defaults(func=cmd_eval)

    p_run = sub.add_parser("run", help="run one input through a prompt")
    p_run.add_argument("task", help="path to a task .yaml / .json")
    p_run.add_argument("--input", required=True, help="the input to run")
    p_run.add_argument("--prompt-file", default=None, help="use this prompt instead of the seed")
    p_run.set_defaults(func=cmd_run)

    sub.add_parser("version", help="print version").set_defaults(
        func=lambda _a: (print(f"loopr {__version__}"), 0)[1]
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
