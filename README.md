# sreval

[![CI](https://github.com/fsantibanezleal/CAOS_SREVAL/actions/workflows/ci.yml/badge.svg)](https://github.com/fsantibanezleal/CAOS_SREVAL/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An honest evaluator for symbolic regression: triple-equivalence testing that reports its own failure rate.

## Why this exists

Symbolic regression is supposed to find the equation, not merely a function that fits. The published
benchmark results show those are very different achievements: a method can score above 0.999 on a
coefficient of determination while recovering the correct expression **structure zero percent of the
time**. An evaluation that reports only accuracy cannot tell a discovery from a good curve fit.

Testing structure directly is the fix, but no single structural test is reliable:

| Test | Strength | Failure mode |
|---|---|---|
| **Symbolic** simplification | Exact when it terminates | Times out, throws, or fails to reduce a genuinely zero difference. Scoring those as method failures charges the search for a defect in the scorer. |
| **Numerical** probing | Robust and cheap | Cannot distinguish "equivalent everywhere" from "agrees on the region we sampled", which is exactly what breaks under extrapolation. |
| **Structural** edit distance | The only one giving **graded** credit | Sensitive to algebraically equivalent rewrites that a human would call the same answer. |

So `sreval` runs all three, reports them **separately**, reports whether they **agreed**, and
publishes the **symbolic-test failure rate** as a first-class number. That last one is the
contribution: it is a property of the measurement apparatus that the field currently absorbs
silently into the scores of the methods being measured.

## Install

```bash
pip install sreval               # numpy only
pip install "sreval[symbolic]"   # adds sympy, needed for the symbolic test
```

## Use

```python
import numpy as np
from sreval import check, summarise

verdict = check(
    candidate_infix="x*(a + b)",
    truth_infix="x*a + x*b",
    variables=["x", "a", "b"],
    candidate_fn=lambda X: X[:, 0] * (X[:, 1] + X[:, 2]),
    truth_fn=lambda X: X[:, 0] * X[:, 1] + X[:, 0] * X[:, 2],
    box=[(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
    candidate_tokens=["mul", "v0", "add", "v1", "v2"],
    truth_tokens=["add", "mul", "v0", "v1", "mul", "v0", "v2"],
)

verdict.recovered              # True: an algebraic rewrite of the same expression
verdict.agreed                 # True: the symbolic and numerical tests concur
verdict.structural.distance    # graded and non-zero: written differently

report = summarise([verdict])
report.symbolic_failure_rate   # the number nobody else publishes
report.notes                   # plain-language statements of what the rates mean
```

## Accuracy and recovery are reported separately, always

```python
from sreval import accuracy_solution, description_length
```

`accuracy_solution` is named at length on purpose. An accuracy solution is not a recovery, and the
**gap between those two rates** is the measurement this package was built to expose. Nothing in the
API merges them for you.

`description_length` is the recommended rule for picking one point off a Pareto front. Selecting the
most accurate member reproduces the field's headline failure, because the most accurate member of a
front is routinely the most over-parameterised one.

## Scope, honestly

- It does **not** perform symbolic regression. It evaluates results from any engine.
- It does **not** decide whether a discovered equation is true. Fitting is not discovering, and no
  equivalence test changes that.
- The structural distance is a **sequence** edit distance over the pre-order traversal, not a full
  tree edit distance. The former is O(n*m), the latter O(n^2 m^2), and at the expression sizes
  symbolic regression actually produces the two agree closely enough that the extra cost buys
  nothing. This is stated because the number is reported, and a reported metric with an implicit
  definition is not reproducible.

## Provenance

Written from published specifications. The widely used reference implementation of this evaluation
protocol is GPL-3.0 licensed and **no part of it is reproduced here**, which is what allows this
package to be MIT.

Built for SymLab, a public research lab on symbolic regression, and extracted as a standalone
package because the evaluation problem is not specific to that lab.

Developed by Felipe Santibanez-Leal.
