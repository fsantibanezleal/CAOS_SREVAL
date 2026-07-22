"""Tests for the three equivalence tests, and for the disagreements between them.

The disagreement tests matter most. This package's argument is that any single structural test is
unreliable, and that argument is only credible if the failure modes are demonstrated rather than
asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from sreval import equivalence as eq
from sreval import metrics


def test_symbolic_recognises_an_algebraic_rewrite():
    """x*(a+b) and x*a + x*b are the same expression written two ways."""
    result = eq.symbolic_equivalent("x*(a + b)", "x*a + x*b", ["x", "a", "b"])
    assert result.decided
    assert result.equivalent is True


def test_symbolic_recognises_a_genuine_difference():
    result = eq.symbolic_equivalent("x + 1", "x + 2", ["x"])
    assert result.decided
    assert result.equivalent is False


def test_symbolic_failure_is_reported_as_undecided_not_as_non_equivalence():
    """A scorer failure must never be reported as a method failure.

    This is the central claim of the package: the common practice charges the search for a defect in
    the scoring tool.
    """
    result = eq.symbolic_equivalent("this is not an expression (((", "x", ["x"])
    assert not result.decided
    assert result.equivalent is None
    assert result.error


def test_numerical_agrees_on_an_identity():
    result = eq.numerical_equivalent(
        lambda X: X[:, 0] * 2.0, lambda X: X[:, 0] + X[:, 0], [(-3.0, 3.0)]
    )
    assert result.decided and result.equivalent is True
    assert result.max_relative_error < 1e-12


def test_numerical_detects_a_difference():
    result = eq.numerical_equivalent(
        lambda X: X[:, 0], lambda X: X[:, 0] * 1.01, [(1.0, 3.0)]
    )
    assert result.decided and result.equivalent is False
    assert result.max_relative_error > 1e-3


def test_numerical_counts_skipped_points_rather_than_hiding_them():
    """An expression undefined across part of the box has said something; averaging over the rest
    would hide it."""
    result = eq.numerical_equivalent(
        lambda X: np.where(X[:, 0] > 0, X[:, 0], np.nan),
        lambda X: X[:, 0],
        [(-1.0, 1.0)],
    )
    assert result.n_points_skipped > 0
    assert result.n_points_compared > 0


def test_numerical_agrees_on_the_box_while_differing_outside_it():
    """The exact failure mode that makes a single numerical test insufficient."""
    inside = eq.numerical_equivalent(
        lambda X: X[:, 0], lambda X: X[:, 0] + (X[:, 0] - 1.0) ** 3, [(0.999, 1.001)]
    )
    outside = eq.numerical_equivalent(
        lambda X: X[:, 0], lambda X: X[:, 0] + (X[:, 0] - 1.0) ** 3, [(5.0, 10.0)]
    )
    assert inside.equivalent is True
    assert outside.equivalent is False


def test_structural_distance_is_zero_for_identical_sequences():
    result = eq.structural_distance(["add", "v0", "C"], ["add", "v0", "C"])
    assert result.distance == 0.0


def test_structural_distance_is_graded_not_binary():
    """The property no binary test has: a nearly-right answer scores better than a wrong one."""
    near = eq.structural_distance(["add", "mul", "v0", "v1", "C"], ["add", "mul", "v0", "v1", "v2"])
    far = eq.structural_distance(["sin", "v0"], ["add", "mul", "v0", "v1", "C"])
    assert 0.0 < near.distance < far.distance <= 1.0


def test_structural_distance_is_bounded():
    result = eq.structural_distance(["a"], ["b", "c", "d", "e", "f"])
    assert 0.0 <= result.distance <= 1.0


def test_check_reports_agreement():
    verdict = eq.check(
        candidate_infix="x*2", truth_infix="x + x", variables=["x"],
        candidate_fn=lambda X: X[:, 0] * 2, truth_fn=lambda X: X[:, 0] + X[:, 0],
        box=[(-2.0, 2.0)],
        candidate_tokens=["mul", "v0", "C"], truth_tokens=["add", "v0", "v0"],
    )
    assert verdict.agreed
    assert verdict.recovered


def test_check_falls_back_to_numerical_when_the_simplifier_fails():
    verdict = eq.check(
        candidate_infix="unparseable ((", truth_infix="x", variables=["x"],
        candidate_fn=lambda X: X[:, 0], truth_fn=lambda X: X[:, 0],
        box=[(-1.0, 1.0)], candidate_tokens=["v0"], truth_tokens=["v0"],
    )
    assert not verdict.symbolic.decided
    assert verdict.recovered is True
    assert "failure rate" in verdict.note


def test_summarise_publishes_the_symbolic_failure_rate():
    """The number this package exists to report."""
    good = eq.check(
        candidate_infix="x", truth_infix="x", variables=["x"],
        candidate_fn=lambda X: X[:, 0], truth_fn=lambda X: X[:, 0],
        box=[(-1.0, 1.0)], candidate_tokens=["v0"], truth_tokens=["v0"],
    )
    broken = eq.check(
        candidate_infix="((((", truth_infix="x", variables=["x"],
        candidate_fn=lambda X: X[:, 0], truth_fn=lambda X: X[:, 0],
        box=[(-1.0, 1.0)], candidate_tokens=["v0"], truth_tokens=["v0"],
    )
    report = eq.summarise([good, broken])
    assert report.n == 2
    assert report.symbolic_failed == 1
    assert report.symbolic_failure_rate == pytest.approx(0.5)
    assert any("simplifier" in note for note in report.notes)


def test_summarise_on_an_empty_list_does_not_divide_by_zero():
    report = eq.summarise([])
    assert report.n == 0 and report.exact_recovery_rate == 0.0


# ------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------


def test_accuracy_solution_and_recovery_are_different_questions():
    """The finding that motivates this package, reproduced as an executable demonstration.

    A degree-5 polynomial fitted to exp(x) over a narrow interval clears the accuracy threshold
    comfortably. It is still the wrong FORM: it has no exponential in it, it will diverge outside
    the fitted range, and its structural distance from the truth is large. A benchmark reporting
    only accuracy would score this a success.
    """
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(1.0, 2.0, size=400))
    truth = np.exp(x)
    coefficients = np.polyfit(x, truth, deg=5)
    surrogate = np.polyval(coefficients, x)

    r2 = metrics.coefficient_of_determination(truth, surrogate)
    assert r2 is not None and r2 > 0.999
    assert metrics.accuracy_solution(truth, surrogate), "clears the accuracy threshold"

    structural = eq.structural_distance(
        ["add", "mul", "C", "square", "v0", "add", "mul", "C", "v0", "C"],
        ["exp", "v0"],
    )
    assert structural.distance > 0.5, "and is structurally the wrong expression"

    # And the give-away: it fails badly outside the interval it was fitted on.
    outside = np.array([4.0, 5.0])
    assert abs(np.polyval(coefficients, outside)[-1] - np.exp(5.0)) / np.exp(5.0) > 0.1


def test_description_length_prefers_the_simpler_of_two_equal_fits():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    simple = metrics.description_length(
        n_nodes=3, n_constants=1, y=y, y_pred=y, n_primitives=8, n_variables=2
    )
    padded = metrics.description_length(
        n_nodes=9, n_constants=4, y=y, y_pred=y, n_primitives=8, n_variables=2
    )
    assert simple.total < padded.total


def test_description_length_components_sum_to_the_total():
    y = np.linspace(0.0, 1.0, 50)
    dl = metrics.description_length(
        n_nodes=5, n_constants=2, y=y, y_pred=y * 1.01, n_primitives=8, n_variables=1
    )
    assert dl.total == pytest.approx(dl.structure + dl.constants + dl.residuals)


def test_pareto_front_keeps_only_non_dominated_points():
    assert metrics.pareto_front([(1.0, 3.0), (0.5, 5.0), (0.6, 7.0), (0.1, 9.0)]) == [0, 1, 3]


def test_r2_returns_none_when_nothing_is_finite():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.coefficient_of_determination(y, np.array([np.nan] * 3)) is None
