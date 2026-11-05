# Contributing to Sparehour

Thank you for your interest in the project. This document covers how to set up a
development environment, run tests, and prepare changes for review.

## Development setup

The project uses Python 3.11 or newer and a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

`ffprobe` is required when a workload uses the `media_probe` artifact validator.

## Running tests

The test suite should pass before any change is submitted:

```bash
.venv/bin/pytest -q
```

To also verify that the package builds cleanly:

```bash
.venv/bin/pip install build
.venv/bin/python -m build
```

## Code style

- Keep changes focused and atomic.
- Match the existing style in the files you touch.
- Do not include credentials, API keys, tokens, or private infrastructure
  details (hostnames, organization names, or image digests from your own
  deployments) in commits or pull requests.
- Workload routes and executor code are the trust boundary; job payloads must
  remain schema-bounded and must not select arbitrary code, models, container
  images, or workflows.

## Model packs and routes

New workload routes must pin exact model IDs, revisions, and digests. Do not
allow job payloads to select model files, LoRAs, or container images at runtime.

## Pull requests

1. Open an issue first for large or architectural changes.
2. Include tests that demonstrate the bug or feature.
3. Ensure the full test suite passes.
4. Update relevant documentation if the user-facing behavior changes.

## License

By contributing, you agree that your contributions will be licensed under the
GNU General Public License v3.0 or later (GPL-3.0-or-later).
