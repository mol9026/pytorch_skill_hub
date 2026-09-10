# test-modification-advisor

本地更新的硬件解耦 PyTorch 测试代码修改助手（Claude Skill）。对比一个测试文件与两套目标机制的差距，一次输出**合并的修改建议报告 + 可直接应用的代码**：

- **能力门控**（`@requires_capabilities` / `Capability` / `_capabilities()`；判据见 `../capability-migrator/references/flag-capability-map.md` §一/§二和 `../capability-migrator/references/capability-spec.md` §三~§六）
- **hw_classification**（`HardwareClassification.GENERIC` / `ACCELERATOR` / `CPU` / `CUDA` / `MPS` / `XPU` 硬件分类标注）

支持三种模式：

- **modify（指导）**：给文件路径 → 输出修改建议报告（发现清单 + 差距分析 + 逐条修改动作 + before→after 代码）。
- **execute（执行）**：明确要求直接修改 → 落盘修改，并通过 `pytorch-commit-pre-check` 运行 lintrunner 与本地测试。
- **review（检视）**：给 PR 链接或本地文件 → 输出检视结论（两套机制是否都符合、不符合的怎么改）。

## 快速使用

从 gitcode 安装三个子 skill 和本 skill：
```bash
git clone https://gitcode.com/ken0yuan/hw-classification-annotator.git
git clone https://gitcode.com/ken0yuan/capability-migrator.git
git clone https://gitcode.com/ken0yuan/pytorch-commit-pre-check.git
git clone https://gitcode.com/ken0yuan/test-modification-advisor.git
```

随后将本skill安装到claude的skill中，随后直接对 Claude 说："用 test-modification-advisor 分析 `/path/to/test_xxx.py`中的XXX，给出修改建议（能力注册 + hw_classification）"。


## 目录结构

```
test-modification-advisor/
├── SKILL.md                       # 精简入口：三种模式流程与规则指针
├── README.md                      # 本文件
├── flowchart.md                   # 规则与 modify/execute/review 流程图
├── scripts/
│   └── scan_combined.py           # 合并扫描：调用两个外部 scan + 渲染交叉点（仅 stdlib）
├── references/
│   ├── coordination-guide.md      # 两机制交互规则（编辑顺序/冲突协调/合并检查清单）
│   └── coverage-preservation.md   # 覆盖守恒
└── templates/
    └── modification-report.md     # 修改建议报告模板（含 before→after 代码块）
```

## 依赖的外部 skill

通过**相对路径**调用三个外部 skill，不复制其规则，建议四个 skill 放在同一父目录下：

| 依赖 | 调用点 | 判据 |
|---|---|---|
| `../capability-migrator` | `scripts/scan_flags.py`（能力扫描） | `../capability-migrator/references/flag-capability-map.md` §一/§二；`../capability-migrator/references/capability-spec.md` §三~§七；示例见 `../capability-migrator/references/migration-examples.md`。注：标准来自[PR #191919](https://github.com/pytorch/pytorch/pull/191919) |
| `../hw-classification-annotator` | `scripts/scan_hw_classification.py`（hw 扫描） | `../hw-classification-annotator/references/hw-signal-map.md` §一；`../hw-classification-annotator/references/hw-classification-spec.md` §二~§六 |
| `../pytorch-commit-pre-check` | execute 模式的 lint 与本地测试 | `../pytorch-commit-pre-check/SKILL.md`；advisor 只维护衔接，不复制预检测规则 |

三个外部 skill 默认与本目录**同级**。外部 scan 脚本缺失时，`scan_combined.py` 标记"外部 scan 不可用"，流程降级为手动检视；预检测 skill 缺失时，execute 模式明确报告验证未完成。

**非同级部署**：可用环境变量覆盖 scan 脚本路径（未设置时回退到默认同级路径）：

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `PYTORCH_TEST_SKILLS_DIR` | 两个扫描 skill 所在的**父目录** | 本 skill 的父目录 |
| `CAPABILITY_MIGRATOR_SCAN` | 能力 scan 脚本的**完整路径** | `<父目录>/capability-migrator/scripts/scan_flags.py` |
| `HW_CLASSIFICATION_ANNOTATOR_SCAN` | hw scan 脚本的**完整路径** | `<父目录>/hw-classification-annotator/scripts/scan_hw_classification.py` |

注意：环境变量只解耦**扫描脚本调用**；判据与预检测仍按相对路径定位。因此「非同级部署」仅用于扫描路径解耦，仍建议四个 skill 同级。

## scan_combined.py 检测什么

合并两份确定性扫描（只检测，不做判断）：
- **能力门控**（scan_flags.py）：散点 flag（`PLATFORM_SUPPORTS_*`、`SM*OrLater`、`IS_SM*`、`TEST_*`、`HAS_*`、`has_*()`、`requires_*`）、skip/xfail 装饰器、写死设备、测试类基类、`instantiate_device_type_tests` 调用、已迁移标记。
- **hw_classification**（scan_hw_classification.py）：每类基类、标注状态、测试方法（device 参数 + 装饰器）、instantiate 调用（only_for/except_for）、硬编码设备引用、**约束检查摘要**。
- **交叉点**：两机制都要核对的项（硬编码设备候选 / instantiate / 迁移与标注状态）。

## 判据入口

- 硬件信号：`../hw-classification-annotator/references/hw-signal-map.md` §一；分类与拆类：`../hw-classification-annotator/references/hw-classification-spec.md` §二/§三；目标形态与声明：同文件 §四/§五。
- flag/SM 语义：`../capability-migrator/references/flag-capability-map.md` §一/§二；普通/分布式迁移、注册与用法：`../capability-migrator/references/capability-spec.md` §三~§六。
- 本 skill 只维护两机制的编辑顺序、交互冲突和覆盖守恒，不复制上述速查表。

## 典型配合（两机制协调）

```python
class TestFlopCounter(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR   # hw：类级"我跨加速器"

    @requires_capabilities(Capability.dtype.fp8)            # capability：方法级"我要 fp8"
    def test_scaled_mm(self, device):                        # device-type 形态使用注入 device
        ...  # device=device，无硬编码

instantiate_device_type_tests(TestFlopCounter, globals(), except_for=("cpu",))
```

## 边界与豁免

- **新能力注册落点**：按 `../capability-migrator/references/capability-spec.md` §五，未注册能力（`Capability` 常量 + 各后端 `_capabilities()`）改的是 `common_device_type.py` / 各 `*TestBase`，不是被迁测试文件——报告里明确标注。
- **`test/cpython` 目录**：linter 显式跳过，标注应在 `CPythonTestCase` 基类上完成。
- **纯逻辑/无设备测试**：不迁能力门控，hw 标 `GENERIC`。

## 其他工具
[#190173](https://github.com/pytorch/pytorch/pull/190173)提供了一种hw-classification的代码级测试，在linter阶段扫描test代码是否有不符合hw-classification标准的代码。目前很多文件处于白名单中未被扫描，如果需要扫描自己修改的文件可以将其从白名单中剔除，达到和第一个子skill检查同样的效果。

## 环境

advisor 的扫描脚本只需 Python 3 标准库，无需安装 torch。execute 模式的环境要求只查 `../pytorch-commit-pre-check/SKILL.md` Step 0，不在这里维护副本。
