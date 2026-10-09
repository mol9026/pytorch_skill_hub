# PyTorch Yongli Skill

This project unifies several decoupled PyTorch skill modules contributed by different team members into one discovery and dispatch layer.

## Integrated skills

The current registry includes nine skill modules under the top-level `skills/` directory:

- `pr-review`
- `pytorch-test-refactoring`
- `test-modification-advisor`
- `torch-npu-test-failure-diagnose`
- `npu-precision-toolkit` — NPU 图模式精度排查四件套（watchdog/doctor/kernel-isolate/alias-tracker）+ 13 模型穿刺模板 + 缺陷指纹库
- `puncture-model-verify` — 模型穿刺验证流程（build_args 契约 + 判定规则 + 台账格式；工具实体在 npu-precision-toolkit）
- `red-green-validation` — 修复隔离验证协议（提 PR 前的红绿灯判定 + verdict 落盘）
- `server-workspace-hygiene` — NPU 服务器工作区整理与本地备份（巡检/归档/备份/双端同步）
- `skill-harvest` — 工作成果归集判定与 skill 脚手架（团队沉淀入口）

Plus one runtime-layer skill exposed via a thin shell for discovery:
- `usage-intelligence` — Claude Code 会话用量与成本优化（实现在 `.claude/skills/`，本目录为团队发现层薄壳）

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
