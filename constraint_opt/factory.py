from __future__ import annotations

from typing import Any, Callable, Dict

# Encoder signature: (domain, traces, objectives) -> encoding instance
ProblemBuilder = Callable[[Any, Any, Any], Any]

_REGISTRY: Dict[str, ProblemBuilder] = {}

def register_problem(problem_type: str, problem_builder: ProblemBuilder) -> None:
    pt = problem_type.lower()
    if pt not in _REGISTRY:
        _REGISTRY[pt] = problem_builder

def resolve(problem_type: str) -> ProblemBuilder:
    pt = problem_type.lower()
    if pt not in _REGISTRY:
        raise ValueError(f"Unknown problem_type={problem_type!r}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[pt]