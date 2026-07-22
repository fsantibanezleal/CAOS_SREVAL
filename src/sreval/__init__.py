"""sreval: an honest evaluator for symbolic regression.

The package exists because of one published result: a method can score above 0.999 on a coefficient
of determination while recovering the correct expression STRUCTURE zero percent of the time. If a
benchmark reports only accuracy, it cannot tell a discovery from a good fit.

What this package adds over reporting accuracy:

- `equivalence.check` runs three independent tests (symbolic, numerical, structural) and reports
  their DISAGREEMENTS rather than resolving them.
- `equivalence.summarise` publishes the **symbolic-test failure rate** as a first-class number. That
  rate is a property of the scoring tool, not of the method being scored, and the common practice of
  absorbing it into method scores systematically penalises whichever method produces expressions the
  simplifier finds hard.
- `metrics` provides the accuracy-solution and complexity measures kept SEPARATE from recovery, so a
  report can never quietly present one as the other.

MIT licensed and written from published specifications. The reference implementation of this
protocol is GPL-3.0 and no part of it is reproduced here.
"""
from __future__ import annotations

from . import equivalence, metrics
from .equivalence import (
    NumericalResult,
    Report,
    StructuralResult,
    SymbolicResult,
    Verdict,
    check,
    numerical_equivalent,
    structural_distance,
    summarise,
    symbolic_equivalent,
)
from .metrics import (
    accuracy_solution,
    coefficient_of_determination,
    description_length,
    normalised_mse,
    pareto_front,
)

__version__ = "0.01.000"

__all__ = [
    "equivalence", "metrics",
    "check", "summarise", "symbolic_equivalent", "numerical_equivalent", "structural_distance",
    "Verdict", "Report", "SymbolicResult", "NumericalResult", "StructuralResult",
    "accuracy_solution", "coefficient_of_determination", "normalised_mse",
    "description_length", "pareto_front",
    "__version__",
]
