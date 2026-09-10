# coordination-guide.md — 两机制合并修改的协调规则

> **本文件解决什么问题**：一份文件要同时满足能力门控与 `hw_classification` 时，如何确定顺序、合并共同改动并处理冲突。
> **领域判据权威源**：flag/SM 语义查 `../../capability-migrator/references/flag-capability-map.md` §一/§二；能力迁移、分布式、注册、装饰器与检查清单查 `../../capability-migrator/references/capability-spec.md` §三~§七；硬件信号查 `../../hw-classification-annotator/references/hw-signal-map.md` §一；分类、拆类、目标代码形态、声明与检查清单查 `../../hw-classification-annotator/references/hw-classification-spec.md` §二~§六。本文件不复制这些规则。
> 覆盖守恒查 `coverage-preservation.md` §一及第 1~7 节。

## 一、rule 1：分工与权威源

| 编号 | 规则 |
|---|---|
| Rule 1.1 | 能力门控回答方法“需要什么能力”；`hw_classification` 回答类“属于什么硬件类型”。两者分别由对应 subskill 判定 |
| Rule 1.2 | subskill 给出的已注册、建议新增、非能力、待定或 `UNRESOLVED` 结论原样使用；advisor 不重新命名、定性或覆盖 |

## 二、rule 2：按迁移后的最终状态分类

| 编号 | 规则 |
|---|---|
| Rule 2.1 | 先确定哪些方法走 capability 迁移、最终使用哪种执行形态，再对迁移完成后的方法集做 hw 分类；不得按半成品状态分类 |
| Rule 2.2 | 逐方法设备集合、拆类条件和分类值全部按 `../../hw-classification-annotator/references/hw-classification-spec.md` §二/§三；advisor 只负责把结果回流到实例化与能力动作 |

一个文件可能处于“方法尚未增加 `device`、类尚未实例化”的中间状态。先定最终方法集，再调用 hw 决策树；证据不足时沿用 hw subskill 的 `UNRESOLVED`，不得猜测分类。

## 三、rule 3：合并编辑顺序

| 编号 | 规则 |
|---|---|
| Rule 3.1 | 顺序：记录覆盖基线 → 定迁移范围 → 按最终状态分类 → 做 capability 动作 → 统一执行范围 → 清理 → 补 hw 声明 → 验证覆盖 |
| Rule 3.2 | `only_for`/`except_for` 只定一次：hw subskill 决定目标形态约束，capability subskill 提供设备 flag 的原始范围 |
| Rule 3.3 | 两个 scan 报出的同一硬编码设备候选只处理一次；是否修改按 `../../hw-classification-annotator/references/hw-classification-spec.md` §三/§四及 `../../capability-migrator/references/capability-spec.md` §三判断 |

执行顺序：

```
⓪ 建覆盖基线      → before device/backend/参数/gate/test ID 与证据状态
① 定迁移范围      → ../../capability-migrator/references/flag-capability-map.md §一/§二 + ../../capability-migrator/references/capability-spec.md §三/§四
② 定 hw 分类      → 最终状态 + ../../hw-classification-annotator/references/hw-classification-spec.md §二/§三；证据不足保留 UNRESOLVED
③ 做能力动作      → ../../capability-migrator/references/capability-spec.md §三~§六
④ 统一执行范围    → ../../hw-classification-annotator/references/hw-classification-spec.md §四 + ../../capability-migrator/references/flag-capability-map.md §一 Rule 1.3/1.6
⑤ 清理共同改动    → 同一候选只改一次
⑥ 补分类声明      → ../../hw-classification-annotator/references/hw-classification-spec.md §五
⑦ 验覆盖守恒      → ./coverage-preservation.md §一及第 1~7 节
```

## 四、rule 4：冲突处理

| 编号 | 规则 |
|---|---|
| Rule 4.1 | 设备范围只保留一个来源：设备专属类已有匹配的 `only_for` 时，等价设备可用性 gate 不再以方法装饰器重复表达 |
| Rule 4.2 | hw 判为纯逻辑 `GENERIC`、而 capability 机制缺少可求值 harness 时，保留原 skip 语义，不强塞能力门控 |
| Rule 4.3 | 两个 subskill 的结论无法同时落地时，报告“需拆类 / 需确认”并列出冲突证据，不得弱化精确 backend、猜设备范围或静默选择一边 |
| Rule 4.4 | 分布式类先按 `../../hw-classification-annotator/references/hw-classification-spec.md` §四“分布式形态”选择 device-type 或 direct-multiprocess，再按 `../../capability-migrator/references/capability-spec.md` §四 C1/C2 实现对应前置检查；advisor 不复述领域细则 |
| Rule 4.5 | 能力装饰位置完全按 `../../capability-migrator/references/capability-spec.md` §六 Rule 6.2，不在 advisor 维护镜像规则 |

补充交互原则：

- 两个 scan 不一致时，scan 结果只作为当前状态的发现清单；目标形态以最终迁移意图和两个 subskill 的权威判据为准。
- 硬编码设备默认改为注入设备；真正测试设备专属 API、错误语义或不同构造路径的候选，按 `../../hw-classification-annotator/references/hw-classification-spec.md` §三第 3 条及 §四 Rule 3.6 的例外保留并在报告说明。
- 覆盖扩大和缩小都必须按 `./coverage-preservation.md` §一 Rule 5.1~5.5 解释；静态计划、collection、runtime 与 `unresolved` 不得混写。

## 五、交互点矩阵

| 交互点 | subskill 输入 | advisor 只负责 |
|---|---|---|
| 硬编码设备候选 | capability 检查注入设备是否控制完整执行链；hw 判断目标路径及语义例外 | 合并为一次修改或一次保留说明 |
| `only_for`/`except_for` | capability 给出设备 flag 的语义范围；hw 给出目标形态约束 | 产出一组同时满足两者的参数 |
| device-type 与 direct multiprocess | `../../hw-classification-annotator/references/hw-classification-spec.md` §四“分布式形态”决定合法绑定；`../../capability-migrator/references/capability-spec.md` §四 C1/C2 决定检查时点与 provider | 选择相容路径；不把 device-type 的参数要求套到 direct multiprocess |
| 待定或冲突证据 | 两个 subskill 各自给出 `UNRESOLVED`/待定及核实条件 | 原样汇总，不代替领域规则下结论 |

## 六、合并检查清单

先分别执行权威检查清单，不在这里复制：

- capability：`../../capability-migrator/references/capability-spec.md` §七。
- hw classification：`../../hw-classification-annotator/references/hw-classification-spec.md` §六。

然后只检查 advisor 独有的交互项：

- [ ] 分类按迁移后的最终方法集判定；hw subskill 的待定或 `UNRESOLVED` 未被 advisor 覆盖
- [ ] `only_for`/`except_for` 已统一定稿，没有设备范围的重复来源
- [ ] 同一硬编码设备候选只处理一次；保留的设备专属语义例外已有依据
- [ ] direct multiprocess 未被套用 device-type 的“每方法 `device` 参数”要求
- [ ] 两个 subskill 无法同时满足的项已报告“需拆类 / 需确认”，未擅自弱化语义
- [ ] 已按 `./coverage-preservation.md` §一及第 1~7 节记录基线、覆盖差异、runtime 证据和未验证项
