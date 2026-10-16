# PyTorch Yongli Skill

This project unifies several decoupled PyTorch skill modules contributed by different team members into one discovery and dispatch layer.

## Integrated skills

The registry currently includes **12 skill modules** under the top-level `skills/` directory, organized by capability domain:

### 测试质量类（团队原有）

- `pr-review` — PR 代码检视
- `pytorch-test-refactoring` — PyTorch 测试用例解耦与重构（含 agent 编排 harness）
- `test-modification-advisor` — 本地测试修改建议与合并修改
- `torch-npu-test-failure-diagnose` — NPU 用例失败根因分析（**只读诊断**：给定用例名 + 日志，不跑实验）

### NPU 图模式精度类（llama2_7b 项目沉淀，2026-10-09 归集）

- `npu-precision-toolkit` — 精度排查四件套（**主动实验**：watchdog 门禁 / doctor 诊断 / kernel 隔离 / buffer 审计）+ 13 模型穿刺模板 + 缺陷指纹库（含 2 个归因修正）。部署需 NPU 服务器
- `puncture-model-verify` — 模型穿刺验证流程（build_args 契约、判定三查、台账格式）；工具实体在 npu-precision-toolkit，本 skill 为流程协议
- `red-green-validation` — 修复隔离验证协议（单撤一修复 → 门禁 → 对照基线 → verdict 落盘）。多修复同时生效时强制使用

### 工程流程类（2026-10-09 归集）

- `server-workspace-hygiene` — NPU 服务器工作区整理与本地备份（巡检/归档/带日期备份/双端同步）
- `npu-fleet-ops` — NPU 机群运维：多机接入（SSH config 别名+公钥分发）、资源审计五件套、磁盘病灶定位、备份双端校验、重装迁移清单
- `skill-harvest` — 工作成果归集判定（三关标准）与 skill 脚手架；维护候选清单

### 会话与成本类

- `model-switch-guard` — 从 CC Switch 网关日志审计模型路由切换（四类事件区分，0 误报回归），估算切换的缓存损失
- `usage-intelligence`（薄壳） — Claude Code 会话用量与成本优化；实现在 `.claude/skills/`，本目录为团队发现层入口，安装方法见其 SKILL.md

> **精度问题的正确入口**：先用 `torch-npu-test-failure-diagnose` 看日志（只读），
> 需要动手验证时转 `npu-precision-toolkit`（跑实验）。两者边界见各自 SKILL.md。

## Project structure

- `src/pytorch_yongli_skill/` : shared Python package for discovery and routing
- `src/pytorch_yongli_skill/registry.py` : registry + local auto-discovery
- `src/pytorch_yongli_skill/__main__.py` : unified CLI entry point
- `skills/` : skill directories kept intact
- `docs/` : manuals — `member-manual.md`（使用手册）、`skills-catalog.md`（**能力目录，按"想做什么"选 skill**）、`leader-summary.md`（汇报材料）、`skilllist.txt`（外部 skill 来源链接）
- `tests/` : validation for discovery, listing, and dispatch behavior

## Unified CLI

```bash
python -m pytorch_yongli_skill --root . list
python -m pytorch_yongli_skill --root . show pr-review
python -m pytorch_yongli_skill --root . read pr-review
python -m pytorch_yongli_skill --root . run pytorch-test-refactoring -- --help
python -m pytorch_yongli_skill --root . route "我需要诊断 NPU 用例失败"
```

> Windows 控制台中文乱码时加 `PYTHONIOENCODING=utf-8`。

## How the integration works

1. Each skill stays in its original directory under `skills/`.
2. The registry scans `skills/*/SKILL.md` and extracts the `name` and `description`.
3. The CLI uses the registry to list, inspect, or dispatch to the matching skill.
4. When a skill has a runnable Python entrypoint, the dispatcher invokes it with forwarded arguments.

## Recommended extension pattern

To add a new skill, place a new directory under `skills/` with a `SKILL.md` file and optional runnable script. The registry will discover it automatically.

**新增 skill 请走 `skill-harvest` 的归集流程**（三关判定 + 标准脚手架 + 验证登记），不要直接堆目录。

## Quick start

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
```
