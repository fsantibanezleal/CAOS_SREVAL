# Changelog

All notable changes are documented here. Format follows Keep a Changelog; newest on top.

## 0.01.000 - 2026-07-22

### Added

- `equivalence.check`, running three independent equivalence tests (symbolic simplification,
  numerical probing across a box, and a normalised structural edit distance) and reporting their
  disagreements rather than resolving them.
- `equivalence.summarise`, publishing the symbolic-test failure rate as a first-class number, on the
  argument that it is a property of the scorer and must not be absorbed into the scores of the
  methods being scored.
- `metrics` with `accuracy_solution`, `coefficient_of_determination`, `normalised_mse`,
  `description_length` and `pareto_front`, keeping accuracy and recovery separate by construction.
- 19 tests, including executable demonstrations of the three failure modes the package argues about:
  a scorer failure reported as undecided rather than as non-equivalence, a numerical test that agrees
  on the sampled box while differing outside it, and a degree-5 polynomial that clears the accuracy
  threshold against exp(x) while being structurally the wrong form and diverging outside the fit.
- PyPI publishing through OIDC trusted publishing, with no stored token.
