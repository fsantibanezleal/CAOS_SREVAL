# Contributing

Issues and pull requests are welcome.

## Ground rules

- **Every claim carries its evidence.** If a change alters a reported number, the pull request states
  the before and after, and how they were measured.
- **A scorer failure is never reported as a method failure.** That distinction is the core of this
  package; a change that blurs it will be declined.
- **No GPL-licensed code.** This package is MIT and is written from published specifications. The
  reference implementation of this protocol is GPL-3.0 and must not be vendored, adapted or
  paraphrased into this repository.

## Development

    python -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    .venv/bin/python -m pytest tests -q
    .venv/bin/python -m ruff check src tests

Branch flow: `task/<issue>/<description>` to `develop`, then `develop` to `main`. Never push `main`.
