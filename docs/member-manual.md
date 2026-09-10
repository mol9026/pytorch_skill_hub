# PyTorch skill 整合使用手册

## 1. 目标

本项目用于统一管理多个 PyTorch 相关 skill，帮助团队成员用一套入口发现、查看和调用各类能力。

## 2. 当前已接入的 skill

- `pr-review`：PR 代码检视
- `pytorch-test-refactoring`：PyTorch 测试用例解耦与重构
- `test-modification-advisor`：本地测试修改建议与合并修改
- `torch-npu-test-failure-diagnose`：NPU 用例故障诊断

## 3. 目录结构

```text
pytorch_yongli_skill/
├── README.md
├── docs/
│   ├── leader-summary.md
│   └── member-manual.md
├── skills/
│   ├── pr-review/
│   ├── pytorch-test-refactoring/
│   ├── test-modification-advisor/
│   └── torch-npu-test-failure-diagnose/
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
pr-review
pytorch-test-refactoring
test-modification-advisor
torch-npu-test-failure-diagnose
```

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

1. 在 `skills/` 下新增一个目录
2. 提供 `SKILL.md`
3. 如有可执行脚本，可继续放在目录内
4. 统一入口会自动发现并注册

## 8. 使用建议

- 对于新成员，先用 `list` 和 `read` 了解整套能力
- 对于实际任务，优先用 `route` 找到最合适 skill
- 对于可执行的 skill，尽量通过统一 `run` 入口发起，而不是直接硬编码目录路径

## 9. 结论

该整合层提供了统一的入口，降低了各 skill 的使用门槛，也为后续扩展更复杂的工作流编排提供了基础。
