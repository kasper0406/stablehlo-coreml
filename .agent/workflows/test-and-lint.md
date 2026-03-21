---
description: How to run tests and lint checks for stablehlo-coreml
---

# Running Tests and Lint

All commands should be run from the repo root `/Volumes/git/stablehlo-coreml`.

## Lint (flake8)

// turbo
```
flake8 . --count --show-source --statistics --max-line-length=127
```

This mirrors the CI check. It catches syntax errors, undefined names, formatting, and line-length violations.

## Core Tests (JAX)

Run the full core test suite (uses `pytest` + coverage via hatch):

// turbo
```
hatch run +py=3.12 test:test-with-cov
```

Run a specific test file:

// turbo
```
hatch run +py=3.12 test:pytest tests/test_jax.py
```

Run a single test by name:

// turbo
```
hatch run +py=3.12 test:pytest tests/test_jax.py::test_gather -v
```

Run tests matching a keyword pattern:

// turbo
```
hatch run +py=3.12 test:pytest tests/test_jax.py -k "test_conv" -v
```

## Equinox Tests

// turbo
```
hatch run +py=3.12 test:pytest tests/test_equinox.py
```

## PyTorch Export Tests

// turbo
```
hatch run +py=3.12 test-pytorch:pytest -vv tests/pytorch/
```

## Notes

- Supported Python versions: `3.12`, `3.13`
- The `test` hatch env includes: pytest, flax, flatbuffers, einops, pillow, equinox, pytest-cov
- The `test-pytorch` hatch env includes: pytest, torch, torchvision, torchax, flax, transformers
