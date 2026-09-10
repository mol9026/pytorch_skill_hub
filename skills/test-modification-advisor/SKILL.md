---
name: test-modification-advisor
description: >-
  当用户要给本地更新的 PyTorch 测试文件出修改建议（能力注册迁移 + hw_classification
  标注合并方案）、直接修改测试并运行 lint 与本地测试，或检视一个 PR 的测试改动是否
  同时符合两套机制时加载。支持指导、执行和检视三种模式，也用于用户问“本地更新了
  测试代码怎么改”“直接改完并验证”或“这个测试文件怎么同时迁到能力注册和加硬件
  分类标注”时。
---

# test-modification-advisor —— 本地更新测试代码的合并修改助手

对同一份测试文件，协调能力门控（`@requires_capabilities`）与硬件分类（`hw_classification`）。三种模式：

- **modify（指导）**：用户给一个本地更新的测试文件（常处于迁了一半的中间状态）→ 输出合并修改报告 + 可直接应用的 before→after 代码。
- **execute（执行）**：用户明确要求直接修改 → 落盘合并修改，再调用 `../pytorch-commit-pre-check` 完成 lintrunner 与本地测试验证。
- **review（检视）**：用户给一个 PR 链接或 diff → 输出检视结论（两套机制是否都符合、不符合怎么改）。

> 判定规则不在这里：flag/SM 语义查 `../capability-migrator/references/flag-capability-map.md` §一/§二，能力迁移、分布式、注册与用法查 `../capability-migrator/references/capability-spec.md` §三~§六；硬件信号查 `../hw-classification-annotator/references/hw-signal-map.md` §一，分类、拆类、目标形态与声明查 `../hw-classification-annotator/references/hw-classification-spec.md` §二~§五。这个 skill 只管**两套规则撞在一起时怎么办**和**迁移有没有偷偷改变测试覆盖**。

## 模式识别

| 输入信号 | 模式 |
|---|---|
| 文件路径 / "怎么改" / "帮我出修改方案" / "本地更新" / "迁了一半怎么补" | modify |
| 文件路径 + “直接改” / “帮我改掉” / “动手改” / “改完验证” | execute |
| PR 链接 / diff 内容 / "检视" / "review" / "对不对" | review |

## modify 流程（给文件，出合并修改方案）

> 每次只分析一个测试文件。合并扫描器不接受目录；需要处理多个文件时，逐文件运行并分别出报告。

**modify-step-1：合并扫描**

跑 `python3 scripts/scan_combined.py <测试文件.py>`，拿到能力发现 + 分类发现 + 两机制交叉点。外部脚本缺失就降级：直接读文件 + 对照两个子 skill 的规则。（Windows 下 `python3` 不可用就改用 `python`。）

**modify-step-2：记录迁移前基线**

按 **rule 5.2** 把现在会跑的测试记成清单；确认不了的写"确认不了"，不猜。

**modify-step-3：定迁移范围**

哪些方法会被迁移（最终有 `device` 参数、被实例化），哪些保持纯逻辑。

**modify-step-4：定分类**

按 **rule 2**（必须以迁移后的状态定，按半成品状态定必错）。

**modify-step-5：能力动作**

按 `../capability-migrator/references/flag-capability-map.md` §一/§二做 flag 分类；按 `../capability-migrator/references/capability-spec.md` §三/§四迁移测试、§五注册新能力、§六检查装饰器用法。

**modify-step-6：统一实例化参数**

按 **rule 3.2 + rule 4**：一次定稿；两套规则撞车按冲突规则处理，消解不了的报"需拆类 / 需确认"。

**modify-step-7：清理**

按 **rule 3.3**：两套机制只出一次修改。

**modify-step-8：补分类声明**

属性 + import + 约束修复，按 `../hw-classification-annotator/references/hw-classification-spec.md` §四（目标代码形态）和 §五（声明形式）。

**modify-step-9：验证覆盖守恒**

按 **rule 5** 三步：对比前后测试清单 → 真跑代表性设备 → 扫漏改。有前后清单时跑 `python3 scripts/compare_collection_manifest.py before.txt after.txt`。

**modify-step-10：出报告**

按模板 `templates/modification-report.md` 出报告，每类附可直接用的 before→after 代码；待核实项、不适用项单列。

## execute 流程（直接修改并验证）

**execute-step-1：合并扫描**

按 modify-step-1 逐个扫描目标测试文件。

**execute-step-2：形成合并修改动作**

按 modify-step-2~8 完成基线、迁移范围、分类、能力动作、实例化参数与清理分析。

**execute-step-3：直接修改并重扫**

把合并修改动作应用到用户授权的文件；待定能力或 `UNRESOLVED` 项不猜测修改。落盘后逐文件重跑合并扫描，再按 modify-step-9 验证覆盖守恒。超出授权范围的必要源码落点只报告，不自动修改。

**execute-step-4：调用提交前预检测**

完整读取并执行 `../pytorch-commit-pre-check/SKILL.md`，提供 PyTorch 仓库绝对路径、本次实际修改的文件列表和用户指定的验证范围；未指定时采用该 subskill 的默认模式。环境门控、PR 文件集、lintrunner、P0/P1 本地测试、失败分类和结果产物均只按该 subskill 的对应步骤执行，不在 advisor 复制或重算。

**execute-step-5：汇总并闭环**

汇总修改、重扫残留、预检测结果与待核实项。与本次修改直接相关且在授权范围内的问题，回到 execute-step-2 修正后重走 execute-step-3~5；其余结果按预检测 subskill 的结论原样报告。

## review 流程（给 PR，判断两套机制是否都符合）

> 检视范围仅限两套机制及其覆盖守恒：能力注册、hw_classification，以及迁移有没有静默改变测试覆盖。与这些无关的格式、CI 问题不在范围内。

每一步发现的问题记入问题清单，最后一步汇总输出。

**review-step-1：读 PR 改动**

以 PR 的改动内容为准，本地文件可能落后。

**review-step-2：查应做未做**

能力侧按 `../capability-migrator/references/flag-capability-map.md` §一/§二扫旧 gate；hw 侧按 `../hw-classification-annotator/references/hw-classification-spec.md` §五 Rule 4.3 查应标未标及 `test/cpython` 豁免。

**review-step-3：查两机制正确性**

能力侧按 `../capability-migrator/references/capability-spec.md` §三~§六，hw 侧按 `../hw-classification-annotator/references/hw-classification-spec.md` §二~§五检查；子 skill 标了"待定"或 `UNRESOLVED` 的，按 `references/coordination-guide.md` §一 Rule 1.2 原样报告。

**review-step-4：查覆盖守恒**

按 **rule 5**。

**review-step-5：输出结论**

一句话结论（✅ 全对 / ⚠️ N 个问题）→ 按类别列问题（分能力 / 分类两维度）+ 怎么改 → 没问题的项最后一句带过。

## 规则速查

| rule | 管什么 | 查哪 |
|---|---|---|
| rule 1 | 分工 | `references/coordination-guide.md` §一 |
| rule 2 | 分类的时机 | `references/coordination-guide.md` §二 |
| rule 3 | 编辑顺序 | `references/coordination-guide.md` §三 |
| rule 4 | 冲突处理 | `references/coordination-guide.md` §四 |
| rule 5 | 覆盖守恒 | `references/coverage-preservation.md` §一 |

执行模式的预检测规则只查 `../pytorch-commit-pre-check/SKILL.md`，不在本 skill 复制。

领域判据的精确入口（不复制到这里）：

- `../capability-migrator/references/flag-capability-map.md` §一/§二；`../capability-migrator/references/capability-spec.md` §三~§七。
- `../hw-classification-annotator/references/hw-signal-map.md` §一；`../hw-classification-annotator/references/hw-classification-spec.md` §二~§六。
