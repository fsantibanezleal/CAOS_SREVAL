# Security policy

## Reporting

Report a vulnerability by opening a private security advisory on the repository, or by email to
fsantibanez@gmail.com. Please do not open a public issue for a vulnerability.

## Scope

`sreval` is a pure computation library with one required dependency (numpy) and one optional one
(sympy). It performs no network access and reads no files.

One caveat worth stating plainly: `symbolic_equivalent` parses expression strings with the sympy
parser. Do not pass untrusted input to it. Expression parsing is not a sandbox, and this package does
not attempt to make it one.
