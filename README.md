# Loopr — a self-improving prompt-optimization loop

**Give Loopr a task prompt and a handful of eval cases. It runs the prompt, scores the
outputs, reflects on the failures in plain language, rewrites the prompt, and repeats —
until it converges on the best-performing version.**

It answers the one question every LLM developer keeps asking by hand: *"which prompt
actually works best for this task?"* — and it answers it as a loop, automatically.

```
🔁 loopr · optimizing 'extract_contact' (scorer=json_field, budget=5, cases=4)

  iter 0: ························    0%  (0/4 cases) *best*
  iter 1: ████████████████████████  100%  (4/4 cases) *best*

✅ stop: converged  |  seed 0% → best 100% (+100%, iter 1)
```

That run is real (local `qwen2.5:14b` via Ollama). The seed prompt — *"Extract info from
this bio."* — produced no parseable JSON, so it scored 0%. Loopr read the failures,
diagnosed the missing format constraint, and rewrote the prompt itself to *"return it in
lowercase as a JSON object with a single field 'role'… Example output: {"role": "backend
engineer"}"* — and hit 100% in one iteration.

---

## Why this product

Prompt and agent **evaluation** is one of the fastest-growing developer-tool categories of
2026 — and the technique that wins is **reflective evolution**, not gradient methods:

- **GEPA** (DSPy's reflective prompt optimizer, ICLR 2026) beats RL (GRPO) by **20%** with
  **35× fewer rollouts**, and lifts MATH accuracy **67% → 93%** from instruction refinement
  alone — no fine-tuning, no few-shot examples.
- The incumbents (MLflow, Confident AI, Opik, LangWatch, Maxim) are heavyweight **eval
  SaaS platforms** — accounts, dashboards, hosted infra.
- Meanwhile the 2026 framework conversation has pivoted hard to **code-first,
  minimal-abstraction runtimes**, and developers reward agents that *learn from failures
  and improve over time*.

**Loopr's wedge: the lightweight, local-first, zero-account version of that idea.** One
`pip install`, a YAML file, and it runs offline on Ollama at zero per-token cost — with a
cloud key as an optional fallback. It is the drop-in self-improving prompt loop, not a
platform you have to adopt.

## How it works

```
        ┌──────────────────────────── the loop ────────────────────────────┐
        │                                                                    │
  seed prompt ─▶ RUN on each case ─▶ SCORE (deterministic) ─▶ best? keep it  │
        ▲                                     │                              │
        │                                     ▼                              │
        └──── REWRITE prompt ◀── REFLECT on the failures (LLM, GEPA-lite) ◀──┘

  stop when: score hits target (converged) · no improvement for N iters (plateau) · budget spent
```

The **scoring and the stop decision are deterministic and eval-gated** — the LLM only
(a) runs the candidate prompt and (b) phrases the reflection. So for a fixed model a run is
reproducible, and a bad reflection can never silently corrupt state (if the rewrite can't be
parsed or drops the `{input}` placeholder, Loopr keeps the current prompt).

## Quickstart

```bash
pip install -e .            # or: pip install loopr
ollama serve                # local backend (or set OPENAI_API_KEY / ANTHROPIC_API_KEY)

loopr optimize loopr/tasks/extract_json.yaml      # evolve the prompt
loopr eval     loopr/tasks/sentiment.yaml         # score the seed prompt once
loopr run      loopr/tasks/sentiment.yaml --input "this is great"
```

`optimize` writes two files to `loopr-out/`: the full iteration **trace** (`*.trace.json`)
and the winning **prompt** (`*.best_prompt.txt`).

### Define a task

```yaml
name: sentiment
description: Classify sentiment as positive / negative / neutral.
seed_prompt: |
  What is the sentiment here?
  {input}
scorer: contains            # exact | contains | regex | json_field | llm_judge
config:
  budget: 6                 # max iterations
  patience: 2               # stop after N non-improving iterations
  target_score: 1.0         # stop early at this mean score
cases:
  - { input: "I love it!", expected: positive }
  - { input: "Worst ever.", expected: negative }
```

The `{input}` placeholder is replaced with each case's input (if absent, the input is
appended). Tasks may be YAML or JSON.

### Use it as a library

```python
from loopr import Task, optimize

result = optimize(Task.from_file("loopr/tasks/sentiment.yaml"))
print(result.best_prompt, result.best_score, result.stop_reason)
```

## Scorers

| scorer       | passes when…                                                        |
|--------------|---------------------------------------------------------------------|
| `exact`      | normalized output equals the expected answer                        |
| `contains`   | expected answer appears in the output (case-insensitive)            |
| `regex`      | a pattern (`scorer_config.pattern` or `expected`) matches           |
| `json_field` | a dotted JSON field (`scorer_config.field`) equals the expected     |
| `llm_judge`  | an LLM grader rules the output correct (optional `scorer_config.rubric`) |

## LLM backends

Local-first, with automatic fallback. Resolution when `LOOPR_PROVIDER=auto` (default):
**Ollama → OpenAI-compatible → Anthropic**.

| env var | purpose |
|---|---|
| `LOOPR_MODEL` | model name (default `qwen2.5:14b`) |
| `OLLAMA_HOST` | Ollama URL (default `http://localhost:11434`) |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | OpenAI-compatible fallback |
| `ANTHROPIC_API_KEY` | Anthropic fallback |
| `LOOPR_PROVIDER` | pin one provider (`ollama` / `openai` / `anthropic`) |

No third-party HTTP dependency — requests go through the standard library. The only runtime
dependency is PyYAML.

## Tests

The deterministic core (scorers, task loading, reflection parsing, and the full loop —
convergence, plateau, budget, and target-score stops) is covered by a pytest suite that runs
with an injected stub LLM, so it needs no network and no model:

```bash
pip install -e ".[dev]"
pytest -q          # 28 passing
```

## License

MIT © Rohit Raj — [rohitraj.tech/agents](https://rohitraj.tech/agents)
