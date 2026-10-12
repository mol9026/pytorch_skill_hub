# PyTorch skill 整合使用手册

> 选 skill 先看 `skills-catalog.md`（按"想做什么"选）；本手册讲怎么装、怎么用统一 CLI。

## 1. 目标

本项目用于统一管理多个 PyTorch 相关 skill，帮助团队成员用一套入口发现、查看和调用各类能力。

## 2. 当前已接入的 skill

- `pr-review`：PR 代码检视
- `pytorch-test-refactoring`：PyTorch 测试用例解耦与重构
- `test-modification-advisor`：本地测试修改建议与合并修改
- `torch-npu-test-failure-diagnose`：NPU 用例故障诊断（只读分析，给定用例名+日志）
- `npu-precision-toolkit`：NPU 图模式精度排查四件套（主动实验：门禁/诊断/隔离/审计 + 穿刺模板）。部署需 NPU 服务器，PC 侧只读结果
- `puncture-model-verify`：模型穿刺验证流程——新模型验证图模式精度的标准协议（args 契约、判定三查、台账格式），工具在 npu-precision-toolkit
- `red-green-validation`：修复定案/提 PR 前的隔离红绿灯验证（单撤一修复 → 门禁 → 对照基线 → verdict 落盘）。多修复同时生效时强制使用
- `server-workspace-hygiene`：服务器目录整理、debug 产物归档、带日期备份到本地、交付文档双端同步
- `skill-harvest`：判断工作沉淀是否应收编进本仓库，并按标准结构生成 skill 草稿
- `model-switch-guard`：从 CC Switch 网关日志审计模型切换（四类事件区分 + 缓存损失估算），装 CC Switch 且多模型混用的同事建议装
- `usage-intelligence`（薄壳）：会话成本优化，安装方法见其 SKILL.md——装到各自项目的 `.claude/` 下

> 精度问题先 `torch-npu-test-failure-diagnose` 看日志（只读），需要动手验证时转 `npu-precision-toolkit`（跑实验）。

## 3. 目录结构

```text
pytorch_yongli_skill/
├── README.md
├── docs/
│   ├── leader-summary.md        # 汇报材料
│   ├── member-manual.md         # 本手册
│   ├── skills-catalog.md        # 能力目录（按"想做什么"选 skill，先看这个）
│   └── skilllist.txt            # 外部 skill 来源链接
├── skills/                      # 11 个 skill，四大类（见 skills-catalog.md）
│   ├── pr-review/                         # 测试质量类
│   ├── pytorch-test-refactoring/
│   ├── test-modification-advisor/
│   ├── torch-npu-test-failure-diagnose/
│   ├── npu-precision-toolkit/             # NPU 精度类（工具层）
│   ├── puncture-model-verify/             #            （流程层）
│   ├── red-green-validation/              #            （定案协议层）
│   ├── server-workspace-hygiene/          # 工程流程类
│   ├── skill-harvest/                     #
│   ├── model-switch-guard/                # 会话与成本类
│   └── usage-intelligence/                # （薄壳，实现在 .claude/skills/）
├── src/
│   └── pytorch_yongli_skill/
│       ├── __init__.py
│       ├── __main__.py
│       └── registry.py
├── tests/
└── pyproject.toml
```

## 4. 基础操作

### 4.1 查看全部 skill

```bash
python -m pytorch_yongli_skill --root . list
```

输出示例：

```text
model-switch-guard
npu-precision-toolkit
pr-review
puncture-model-verify
pytorch-test-refactoring
red-green-validation
server-workspace-hygiene
skill-harvest
test-modification-advisor
torch-npu-test-failure-diagnose
usage-intelligence
```

> 中文 skill 的 show/read 输出乱码时加 `PYTHONIOENCODING=utf-8`。

### 4.2 查看某个 skill 的元信息

```bash
python -m pytorch_yongli_skill --root . show pr-review
```

返回内容包括：

- skill 名称
- description
- entrypoint
- tags
- metadata

### 4.3 阅读某个 skill 的说明文档

```bash
python -m pytorch_yongli_skill --root . read pr-review
```

这会直接输出该 skill 对应的 `SKILL.md` 内容。

### 4.4 运行某个 skill 的可执行入口

```bash
python -m pytorch_yongli_skill --root . run pytorch-test-refactoring -- --help
```

如果一个 skill 有可执行入口（如 `orchestrator.py`），该命令会转发参数并执行它。

## 5. 自动路由能力

如果你不知道该用哪个 skill，可以直接描述需求：

```bash
python -m pytorch_yongli_skill --root . route "我需要诊断 PyTorch-NPU 用例失败"
```

返回值会是最匹配的 skill 名称，例如：

```text
torch-npu-test-failure-diagnose
```

## 6. 适用场景

### 场景 A：想查看某个 skill 的用途

```bash
python -m pytorch_yongli_skill --root . read <skill-name>
```

### 场景 B：不知道具体选哪个

```bash
python -m pytorch_yongli_skill --root . route "你的需求描述"
```

### 场景 C：希望直接执行某个 skill

```bash
python -m pytorch_yongli_skill --root . run <skill-name> -- <pass-through args>
```

## 7. 扩展方式

新增 skill 时，只需按照以下规则：

1. 先过 `skill-harvest` 的三关判定（价值/稳定/归属）
2. 按其标准脚手架建目录 + SKILL.md
3. 统一入口会自动发现并注册
4. 在 skill-harvest 的已归集清单登记

## 8. 使用建议

- 对于新成员，先用 `list` 和 `read` 了解整套能力
- 对于实际任务，优先用 `route` 找到最合适 skill
- 对于可执行的 skill，尽量通过统一 `run` 入口发起，而不是直接硬编码目录路径

## 9. 结论

该整合层提供了统一的入口，降低了各 skill 的使用门槛，也为后续扩展更复杂的工作流编排提供了基础。
