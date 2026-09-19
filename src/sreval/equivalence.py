"""Triple-equivalence testing: is a discovered expression the SAME as the true one?

This is the question symbolic regression exists to answer and the one its benchmarks answer worst.
The published result that motivates this package is stark: a method can score above 0.999 on a
coefficient of determination while recovering the correct structure zero percent of the time. A
perfect fit that is structurally the wrong equation is not a discovery, and only a structural test
can tell the difference.

The complication is that no single structural test is reliable:

- **Symbolic** simplification is exact when it terminates, but on a non-trivial fraction of real
  expressions the simplifier times out, throws, or simply fails to reduce a genuinely zero
  difference to zero. The common practice of scoring those as method failures is wrong: it charges
  the search for a defect in the scoring tool.
- **Numerical** probing is robust and cheap, but it cannot distinguish "equivalent everywhere" from
  "agrees on the region we sampled", which is exactly the distinction that matters for extrapolation.
- **Structural** edit distance is the only one that gives GRADED credit, so a nearly-right expression
  is distinguishable from a completely wrong one. But it is sensitive to algebraically equivalent
  rewrites that a human would call the same answer.

So this package runs all three, reports them separately, reports whether they AGREED, and reports the
symbolic-test failure rate as a first-class number. That last one is the contribution: it is a defect
of the measurement apparatus that the field currently absorbs silently into method scores.

Nothing here is copied from any existing benchmark implementation. The reference implementation of
this protocol is GPL-3.0; this package is MIT and is written from the published specification.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Points drawn for the numerical probe.
DEFAULT_PROBE_POINTS = 512

#: Relative tolerance below which two expressions count as numerically equivalent.
DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SymbolicResult:
    """The symbolic verdict, or an explicit statement that the tool could not decide."""

    decided: bool
    equivalent: bool | None
    error: str = ""
    seconds: float = 0.0


@dataclass(frozen=True)
class NumericalResult:
    """The numerical verdict, with the worst relative deviation observed."""

    decided: bool
    equivalent: bool | None
    max_relative_error: float | None
    n_points_compared: int
    n_points_skipped: int


@dataclass(frozen=True)
class StructuralResult:
    """The graded structural verdict, in [0, 1]. Zero means identical token sequences."""

    distance: float
    n_tokens_candidate: int
    n_tokens_truth: int


@dataclass
class Verdict:
    """The combined outcome of all three tests."""

    symbolic: SymbolicResult
    numerical: NumericalResult
    structural: StructuralResult
    agreed: bool
    note: str = ""

    @property
    def recovered(self) -> bool:
        """The reported verdict: symbolic when it decided, otherwise numerical.

        Deliberately NOT a majority vote. The structural distance is graded rather than binary and
        turning it into a vote would require a threshold that nobody has justified.
        """
        if self.symbolic.decided and self.symbolic.equivalent is not None:
            return bool(self.symbolic.equivalent)
        return bool(self.numerical.equivalent)


def symbolic_equivalent(
    candidate: str,
    truth: str,
    variables: Sequence[str],
    *,
    timeout_seconds: float = 10.0,
) -> SymbolicResult:
    """Simplify `candidate - truth` and test it against zero.

    Expressions are given as infix strings so this package does not impose an expression class on
    its callers. A failure returns `decided=False` with the reason attached; it is never reported as
    a non-equivalence, because those are different claims.
    """
    import time

    started = time.perf_counter()
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr

        symbols = {name: sympy.Symbol(name, real=True) for name in variables}
        a = parse_expr(candidate, local_dict=symbols, evaluate=True)
        b = parse_expr(truth, local_dict=symbols, evaluate=True)
        difference = sympy.simplify(a - b)
        return SymbolicResult(
            decided=True,
            equivalent=bool(difference == 0),
            seconds=round(time.perf_counter() - started, 4),
        )
    except Exception as error:  # noqa: BLE001 - any failure here is a SCORER failure
        return SymbolicResult(
            decided=False,
            equivalent=None,
            error=f"{type(error).__name__}: {error}",
            seconds=round(time.perf_counter() - started, 4),
        )


def numerical_equivalent(
    candidate_fn: Callable[[np.ndarray], np.ndarray],
    truth_fn: Callable[[np.ndarray], np.ndarray],
    box: Sequence[tuple[float, float]],
    *,
    n_points: int = DEFAULT_PROBE_POINTS,
    tolerance: float = DEFAULT_TOLERANCE,
    seed: int = 0,
) -> NumericalResult:
    """Compare two callables at random points across a box.

    Points where either side is non-finite are SKIPPED and counted, never silently dropped: an
    expression that is undefined across a third of the box has told you something, and hiding that
    behind a clean average on the remainder is the failure this package exists to avoid.
    """
    rng = np.random.default_rng(seed)
    probe = np.column_stack([rng.uniform(low, high, size=n_points) for low, high in box])
    a = np.asarray(candidate_fn(probe), dtype=np.float64)
    b = np.asarray(truth_fn(probe), dtype=np.float64)
    usable = np.isfinite(a) & np.isfinite(b)
    n_skipped = int(np.sum(~usable))
    if not usable.any():
        return NumericalResult(False, None, None, 0, n_skipped)
    denominator = np.maximum(np.abs(b[usable]), 1e-12)
    worst = float(np.max(np.abs(a[usable] - b[usable]) / denominator))
    return NumericalResult(
        decided=True,
        equivalent=bool(worst <= tolerance),
        max_relative_error=worst,
        n_points_compared=int(np.sum(usable)),
        n_points_skipped=n_skipped,
    )


def structural_distance(candidate_tokens: Sequence[str], truth_tokens: Sequence[str]) -> StructuralResult:
    """Normalised edit distance between two pre-order token sequences, in [0, 1].

    A sequence edit distance over the pre-order traversal, not a full tree edit distance. The former
    is O(n*m); the latter is O(n^2 m^2) and, at the expression sizes symbolic regression actually
    produces, the two agree closely enough that the extra cost buys nothing. This choice is stated
    because the number is reported, and a reported metric with an implicit definition is not
    reproducible.

    Callers should collapse constant leaves to a single token before calling, so that numeric drift
    is not scored as structural difference.
    """
    a, b = list(candidate_tokens), list(truth_tokens)
    if not a and not b:
        return StructuralResult(0.0, 0, 0)
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if token_a == token_b else 1),
            ))
        previous = current
    distance = previous[-1] / max(len(a), len(b))
    return StructuralResult(round(min(1.0, distance), 6), len(a), len(b))


def check(
    *,
    candidate_infix: str,
    truth_infix: str,
    variables: Sequence[str],
    candidate_fn: Callable[[np.ndarray], np.ndarray],
    truth_fn: Callable[[np.ndarray], np.ndarray],
    box: Sequence[tuple[float, float]],
    candidate_tokens: Sequence[str],
    truth_tokens: Sequence[str],
    seed: int = 0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Verdict:
    """Run all three tests and report whether they agreed."""
    symbolic = symbolic_equivalent(candidate_infix, truth_infix, variables)
    numerical = numerical_equivalent(candidate_fn, truth_fn, box, seed=seed, tolerance=tolerance)
    structural = structural_distance(candidate_tokens, truth_tokens)

    decided = [
        r.equivalent for r in (symbolic, numerical)
        if getattr(r, "decided", False) and r.equivalent is not None
    ]
    agreed = len(set(decided)) <= 1

    note = ""
    if not agreed:
        note = (
            f"the symbolic test says {symbolic.equivalent} and the numerical test says "
            f"{numerical.equivalent}. Either the expressions agree on the sampled box but differ "
            "outside it, or the simplifier failed to reduce an equivalent difference to zero. Both "
            "are informative and neither is resolved here."
        )
    elif not symbolic.decided:
        note = (
            f"the symbolic test could not decide ({symbolic.error}). The numerical verdict is used, "
            "and this run counts towards the reported symbolic-test failure rate."
        )

    return Verdict(symbolic=symbolic, numerical=numerical, structural=structural,
                   agreed=agreed, note=note)


@dataclass
class Report:
    """Aggregate statistics over many verdicts, reported the way this package argues they should be."""

    n: int
    exact_recoveries: int
    exact_recovery_rate: float
    symbolic_attempted: int
    symbolic_failed: int
    symbolic_failure_rate: float
    disagreements: int
    disagreement_rate: float
    mean_structural_distance: float
    notes: list[str] = field(default_factory=list)


def summarise(verdicts: Sequence[Verdict]) -> Report:
    """Aggregate verdicts, keeping the failure rate of the measurement apparatus visible.

    `symbolic_failure_rate` is the number this package exists to publish. It is a property of the
    SCORER, not of the methods being scored, and folding it into a method's score, as the common
    practice does, systematically penalises whichever method happens to produce expressions the
    simplifier finds hard.
    """
    n = len(verdicts)
    if n == 0:
        return Report(0, 0, 0.0, 0, 0, 0.0, 0, 0.0, 0.0)
    recoveries = sum(1 for v in verdicts if v.recovered)
    attempted = n
    failed = sum(1 for v in verdicts if not v.symbolic.decided)
    disagreements = sum(1 for v in verdicts if not v.agreed)
    distances = [v.structural.distance for v in verdicts]
    notes: list[str] = []
    if failed:
        notes.append(
            f"The symbolic test failed to decide on {failed} of {attempted} comparisons "
            f"({failed / attempted:.1%}). Those fell back to the numerical verdict. A benchmark that "
            "scores such cases as method failures is measuring its own simplifier."
        )
    if disagreements:
        notes.append(
            f"The symbolic and numerical tests disagreed on {disagreements} of {attempted} "
            "comparisons. Disagreement usually means agreement on the sampled box only, which is "
            "precisely the situation that breaks under extrapolation."
        )
    return Report(
        n=n,
        exact_recoveries=recoveries,
        exact_recovery_rate=round(recoveries / n, 6),
        symbolic_attempted=attempted,
        symbolic_failed=failed,
        symbolic_failure_rate=round(failed / attempted, 6),
        disagreements=disagreements,
        disagreement_rate=round(disagreements / attempted, 6),
        mean_structural_distance=round(float(np.mean(distances)), 6),
        notes=notes,
    )
