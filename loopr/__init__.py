"""Loopr - a self-improving prompt-optimization loop.

Give Loopr a task prompt and a handful of eval cases. It runs the prompt,
scores the outputs, reflects on the failures in plain language, rewrites the
prompt, and repeats until it converges on the best-performing version.

The scoring and the stop decision are deterministic and eval-gated; the LLM
only runs the task and phrases the reflection. So a run is reproducible given
a fixed model.
"""

from loopr.task import Task, Case, TaskConfig
from loopr.loop import optimize, evaluate, OptimizeResult, Iteration, CaseResult
from loopr.llm import LLMClient, LLMError
from loopr.scorers import get_scorer, SCORERS

__version__ = "0.1.0"

__all__ = [
    "Task",
    "Case",
    "TaskConfig",
    "optimize",
    "evaluate",
    "OptimizeResult",
    "Iteration",
    "CaseResult",
    "LLMClient",
    "LLMError",
    "get_scorer",
    "SCORERS",
    "__version__",
]
