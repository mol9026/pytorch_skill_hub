# coverage-preservation.md — 迁移覆盖守恒验证

> **本文件解决什么问题**：证明迁移没有静默改变测试覆盖（test-modification-advisor 的 modify-step-2 / step-9、review-step-4 用）。
> 只有 **1 个 rule**：§一 rule 5（覆盖守恒）；目录里的 1~7 节是各条 rule 的操作细则。能力领域规则以 `../../capability-migrator/references/flag-capability-map.md` §一/§二和 `../../capability-migrator/references/capability-spec.md` §三~§七为准；硬件领域规则以 `../../hw-classification-annotator/references/hw-signal-map.md` §一和 `../../hw-classification-annotator/references/hw-classification-spec.md` §二~§六为准。

## 一、rule 5：覆盖守恒

先说人话：迁移是"换个写法"，**不是"换一批测试"**。迁移之前会跑 100 个测试，迁移之后还得是这 100 个——不能悄悄变成 95 个，也不能悄悄变成 120 个。这就是覆盖守恒。

| 编号 | 规则 |
|---|---|
| Rule 5.1 | 迁移前后，实际会跑的测试必须还是同一批：哪些测试、跑在哪些设备上、用什么参数，都不能悄悄变多或变少 |
| Rule 5.2 | 怎么确认：迁移前记一份"现在会跑哪些测试"的清单；迁移后再收集一份清单；两份对比，多出来的和消失的每一个都要解释清楚（对比脚本：`scripts/compare_collection_manifest.py`） |
| Rule 5.3 | 测试变多也是问题，不是白赚的。变多变少都要写明理由，不许默认"更多就是更好" |
| Rule 5.4 | 每个结论要注明你是凭什么说的：只是看代码推的 / 拿到测试清单了 / 真跑过了 / 确认不了。**不许把"我看代码觉得对"说成"我验证过了"**（证据等级见第 1 节） |
| Rule 5.5 | 缺目标硬件（比如手里没有 XPU）就老实写"没验证"，不许装通过 |
| Rule 5.6 | 收尾扫一遍有没有漏改的东西：旧 flag / import、多余的装饰器、没用的 device 参数、失效的 skip（见第 6 节） |

## 目录

1. 证据等级（rule 5.4 的细则）
2. 迁移前基线（rule 5.2 的第一步）
3. 迁移后预期矩阵（rule 5.3 的细则）
4. Collection 对比（rule 5.2 的第二步）
5. Runtime 验证（rule 5.5 的细则）
6. 残留与结构检查（rule 5.6 的细则）
7. 专项检查

## 1. 证据等级

对每项结论标记证据状态：

- `static-plan`：只从源码推导，尚未收集或运行。
- `collected`：已有目标 runner 的 test ID/collection 证据。
- `runtime-verified`：至少一个目标设备/backend 实际执行通过。
- `unresolved`：动态表达式、父类 dispatch、缺少设备或依赖导致无法确认。

不得把静态推断写成运行验证。缺少非 CUDA accelerator 时，将 accelerator-agnostic 声明标为 `unresolved`，不能伪装全部通过。

## 2. 迁移前基线

对每个测试方法记录：

| 测试 | device/backend | dtype/参数 | gate/skip | test ID | 证据 |
|---|---|---|---|---|---|

从旧 decorator、实例化调用、参数化、父类和实际 collection 共同确定。无法静态解析的 `only_for`/`except_for` 或 decorator 实参必须保留为 unresolved。

## 3. 迁移后预期矩阵

逐测试写明预期的 device、backend、dtype/参数组合、capability/skip 和 test ID。扩大与缩小覆盖都是行为变化，必须列出理由和证据；不要把“更宽”自动视为正确。

特别检查：

- `HAS_GPU` 等旧 gate 的真实设备集合；
- `only_for` ↔ `except_for` 是否增加或删除 MPS/XPU/PrivateUse1；
- CPU 参数是否被外层 accelerator availability gate 一起跳过；
- device instantiation 与普通参数化是否保持预期组合；
- 拆类、重命名或移动后的 old→new test mapping。

## 4. Collection 对比

在相同 runner、选择参数和环境下保存 before/after test ID manifest，运行：

```bash
python3 scripts/compare_collection_manifest.py before.txt after.txt
```

逐项解释 removed 与 added test ID。确认模板类确实实例化、类名未覆盖、import/discovery error 未被过滤器隐藏、pytest 与 unittest 的选择语义没有意外分叉。

## 5. Runtime 验证

- device-type 迁移至少运行一个非 CPU 实例。
- 声称 accelerator-agnostic 时尽量运行一个非 CUDA accelerator；环境缺失则明确标 unresolved。
- distributed 至少验证一个真实 backend/rank 组合，且 capability 检查发生在 worker spawn/初始化之前。
- 对已知不支持 backend 验证结果为 skip，而不是 import、setup 或执行阶段错误。

记录精确命令、环境、通过/跳过/失败数以及失败原因。

## 6. 残留与结构检查

迁移完成前检查：

- 旧 flag/import、冗余 `@only*`、未使用 `device`、硬编码设备候选；
- 重复测试/helper、失效 skip、未实例化模板类；
- class docstring 仍是类体第一条语句；
- 共享状态通过 context manager 或 `try/finally` 恢复；OOT `_capabilities` monkeypatch 同时恢复原方法并清 cache；
- 语义独立的 shape/dtype/config/backend case 可独立收集；具有状态累积语义的循环不强制拆分。

## 7. 专项检查

### Graph-mode

- 新 Capability/decorator/helper 在目标 merge base 已存在，并在该基线上 collection。
- 测试卫生修改不改变 export DB fixture、`example_args`、预期图结构或输入语义。
- 通用 factory 与设备模块 factory 可能产生不同 Python 类型/tracing 路径；除非有等价证据，保留所有原对象构造路径。

### Distributed

- rank-local device 从统一的 `device_type`/backend resolver 与 `rank`/`local_rank` 派生。
- 精确 backend gate 不弱化为泛化 `distributed.backend`。
- NumPy/SciPy、device count、world-size、dtype 和通信能力等正交 gate 原样保留。
- 拆分后的 CPU/Gloo、CUDA/NCCL 或 accelerator 类共享相同 helper，避免实现漂移。
