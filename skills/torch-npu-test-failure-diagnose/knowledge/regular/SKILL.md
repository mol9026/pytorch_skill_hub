---
name: torch-npu-test-regular-diagnose
description: >-
  当 NPU 用例被判定为非分布式用例时加载。执行步骤 2-5 的常规诊断：
  错误特征提取、根因分类、CANN 可行性判定、级联缺失检测。
---

# 常规用例诊断（步骤 2-5）

> 本文件由根 SKILL.md 在判定为非分布式用例后加载。执行步骤 2-5 的分析，完成后返回根 SKILL.md 输出报告。

### 步骤 2：错误特征提取

从日志中提取以下信息，逐项记录到报告中：

| 提取项 | 搜索方式 | 对应根因类型 |
|--------|----------|-------------|
| 错误类型 | `RuntimeError` / `AssertionError` / `ImportError` / `GradcheckError` | 路由关键信号 |
| 错误消息 | 错误栈中第一条有意义的消息 | 直接指向根因 |
| 调用栈 | 最后 5-10 帧，标注是否包含 `torch_npu`/`op-plugin` 帧 | 区分 NPU 问题 vs 上游问题 |
| fallback 告警 | `will fall back to run on the CPU` / `not currently supported on the NPU backend` | 算子缺失信号 |
| dtype 告警 | `dtype cast replace with float` / `not implemented for DT_DOUBLE` | dtype 支持缺失 |
| CANN 错误码 | `EZ1001`(参数错误) / `EI0004`(OOM) / `ERR00100`(PTA call acl api failed) / `ERR01004`(OPS invalid pointer) | CANN 层面问题 |
| 触发输入 | `SampleInput(input=Tensor[...])` | 复现条件 |

> **补充参考**：
> - `knowledge/regular/routing.md` — 更详细的错误码速查表和 fallback 告警注意事项。当 CANN 错误码不在上述表格中，或需要确认多特征冲突优先级时查阅
> - `knowledge/regular/common-issues.md` — 7 类常见问题的快速参考（算子回退 CPU、接口不存在、类型检查失败等），可用于快速匹配已知问题模式
> - `knowledge/regular/diagnosis-pitfalls.md` — 6 类常见误判陷阱，特别关注陷阱 2（patch 未生效）和陷阱 4（未检查 disabled_testcases）

在根据步骤 2 提取的特征进行优先级路由之前，先将错误日志与已知不支持模式库进行匹配。若命中直接终止，跳过后续步骤。

**快速判定**：对照 `knowledge/common/unsupported-patterns.md` 中 6 类泛化规则，按类别 1→6 顺序匹配错误消息 + SampleInput 行。

```
命中类别 1（dtype 不支持） → ⛔ cann-not-implemented（CANN dtype 缺失）
命中类别 2（硬件维限）     → ⛔ cann-not-implemented（硬件限制）
命中类别 3（CUDA 独有）    → ⛔ cann-not-implemented（NPU 无对标能力）
命中类别 4（格式限制）     → ⛔ cann-not-implemented（内存格式限制）或 → 步骤 3（如含 contiguous 守卫提示）
命中类别 5（白名单算子）   → ⛔ cann-not-implemented（已评估不可组合）
命中类别 6（测试限制）     → ⚠️ 测试框架限制（不可在 torch_npu 侧修复）

全部不命中 → 进入步骤 3 常规优先级路由
```

### 步骤 3：根因分类

根据步骤 2 提取的特征，按冲突优先级路由。详细匹配条件和判定规则见 `knowledge/regular/routing.md`。

```
优先级 1：环境/编译问题 → ✅ 可修复（重配置环境）
优先级 2：上游问题 → ⚠️ 不可在 torch_npu 侧修复（需向上游提 PR 或 workaround）
优先级 3：算子未适配 → 进入步骤 4（CANN 可行性判定）
优先级 4：算子行为不一致 → ✅ 可修复（修正 dtype/shape/实现逻辑）
优先级 5：内存问题 → ⚠️ 有 workaround，但根治可能需要 CANN 侧优化
优先级 6：未分类 → 标记"需人工分析"，列出所有已提取的特征
```

> **补充参考**：
> - `knowledge/regular/routing.md` — 完整路由表、多特征冲突优先级、CANN 错误码速查
> - `knowledge/regular/missing-dispatch.md` 的「排除检查」节 — 当同时出现 fallback 告警和 `aclrtMemcpy`/`copy_`/`Null_Pointer` 帧时，fallback 可能是 side effect
> - `knowledge/regular/upstream-issue.md` — 优先级 2（上游问题）的详细判定步骤和反例

### 步骤 4：CANN 可行性判定（仅"算子未适配"类型）

当根因为"算子未适配"时，判定是"可修复"还是"不可修复（cann-not-implemented）"。完整判定流程见 `knowledge/regular/fixability-rules.md`。

#### 4.1 提取缺失算子名

```
正则：aten::(\w+)
示例：
  "operator 'aten::linalg_householder_product' is not currently supported"
  → 缺失算子：linalg_householder_product
  "call aclnnSort failed ... Input data type should not be float64"
  → 缺失能力：aclnnSort 的 float64 支持
```

#### 4.2 三级判定流程

```
1. CUDA 对照 → CUDA 无原生实现 → ✅ 可修复（fallback 对齐）
                CUDA 有原生实现 → 进入第 2 步

2. 缺失类型分析 → CANN API 不存在（ERR01004）→ 进入第 3 步
                  dtype 不支持（error code 161002）→ 查日志中 supported dtype 列表判定
                  dispatch 缺失（无 aclnn 错误）→ 搜索 NPU 仓库判定

3. 组合可行性 → 可组合 → ✅ 可修复（模式 C2）
                 不可组合 → ⛔ cann-not-implemented（终止）
```

> **⛔ CANN 未实现终止规则**：当 CUDA 有原生实现 + 缺失算子依赖 CANN 底层能力 + CANN 无对应接口 + 无法组合实现，四条件同时满足时终止诊断。详见 `knowledge/regular/fixability-rules.md`。
>
> **补充参考**：
> - `knowledge/regular/fixability-rules.md` — 三级判定详细说明和典型场景速查表
> - `knowledge/regular/missing-dispatch.md` — 排除检查、CANN 终止规则、诊断搜索命令、适配层判断
> - `references/missing-dispatch-guide.md` — 级联缺失处理、反面案例、修复模式

#### 4.3 判定输出

```
可修复 → 根因分析.md 标注"可修复" + 修复方向 + 建议进入 pipeline 实施修复
cann-not-implemented → 根因分析.md 标注"不可修复" + 缺失 CANN 能力 + 建议向 CANN 提交需求
信息不足 → 列出需要确认的事项和对应的搜索命令
```

### 步骤 5：级联缺失检测

如果日志中暴露了多个不同的 `aten::xxx`，或从调用栈可以推断存在级联缺失：

```
检测规则：
  - 同一日志中出现 ≥2 个不同 aten::xxx 缺失 → 标记"级联缺失"
  - 用户提供了多轮 re-run 日志，每轮暴露新算子 → 标记"级联缺失"并列出时间线
```

记录到报告中，提示这是一个级联问题，需要 `missing-dispatch-loop` 循环处理。



---

> 分析完成。返回根 SKILL.md 的步骤 3 输出诊断报告。
