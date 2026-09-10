# PyTorch Yongli Skill

This project unifies several decoupled PyTorch skill modules contributed by different team members into one discovery and dispatch layer.

## Integrated skills

The current registry includes four skill modules under the top-level `skills/` directory:

- `pr-review`
- `pytorch-test-refactoring`
- `test-modification-advisor`
- `torch-npu-test-failure-diagnose`

## Project structure

- `src/pytorch_yongli_skill/` : shared Python package for discovery and routing
- `src/pytorch_yongli_skill/registry.py` : registry + local auto-discovery
- `src/pytorch_yongli_skill/__main__.py` : unified CLI entry point
- `skills/` : original skill directories kept intact
- `tests/` : validation for discovery, listing, and dispatch behavior

## Unified CLI

The project exposes a single entry point for browsing and invoking the integrated skills:

```bash
python -m pytorch_yongli_skill --root . list
python -m pytorch_yongli_skill --root . show pr-review
python -m pytorch_yongli_skill --root . read pr-review
python -m pytorch_yongli_skill --root . run pytorch-test-refactoring -- --help
```

## How the integration works

1. Each skill stays in its original directory under `skills/`.
2. The registry scans `skills/*/SKILL.md` and extracts the `name` and `description`.
3. The CLI uses the registry to list, inspect, or dispatch to the matching skill.
4. When a skill has a runnable Python entrypoint, the dispatcher invokes it with forwarded arguments.

## Recommended extension pattern

To add a new skill, place a new directory under `skills/` with a `SKILL.md` file and optional runnable script. The registry will discover it automatically.

## Quick start

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
```
