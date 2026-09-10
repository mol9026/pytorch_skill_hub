---
name: torch-npu-test-failure-diagnose
description: >-
  Use when the user provides a failed PyTorch-NPU test case name and logs,
  and needs a root cause analysis without modifying code. Classifies the
  failure, determines whether it is fixable within torch_npu scope or
  requires CANN/upstream changes, and outputs a structured diagnosis report.
  当用户需要诊断 PyTorch-NPU 用例失败根因、分析 NPU 算子报错、
  或排查 torch_npu 适配问题时使用。
---

# NPU 用例根因分析（只读诊断）

## 概述

根据用户提供的用例名、执行结果和日志，执行根因分析并输出结构化诊断报告。只读诊断，不修改代码。

## 何时使用

- 用户提供 PyTorch-NPU 用例名 + FAIL 日志，需要根因分析
- 用户需要判断 NPU 算子报错是否可修复（torch_npu 侧 vs CANN 侧 vs 上游）
- 用户需要排查 torch_npu 适配问题（算子缺失、行为不一致、精度差异等）
- 用户提到"诊断 NPU 报错"、"分析算子失败"、"torch_npu 用例挂了"等

## 约束与边界

- **只分析，不修改代码**。不执行任何 `Edit`、`Write`（除报告文件外）、`Bash` 写操作
- **即使结论为"可修复"，也只提出修改建议和修复方向，不直接编写或修改代码**。修复代码由 `torch-npu-issue-pipeline-core` 的实施阶段执行
- **可修复的案例，报告中应包含修复伪代码/代码示例**，供用户和实施阶段参考。但不得直接修改本地机器上的代码文件
- 分析结论必须基于日志证据，不得猜测
- 输出结构化报告：`{workspace}/issues/{case_name}/根因分析.md`
- 如果日志信息不足以做出结论，明确列出需要补充的信息
- 本技能不覆盖修复实施、编译、验证、交付，这些属于 `torch-npu-issue-pipeline-core`

## 输入

用户可以通过对话或粘贴方式提供：

| 输入项 | 必须 | 说明 |
|--------|------|------|
| `case_name` | ✅ | 用例全名，如 `test_fn_grad_sort_npu_float64` |
| 执行结果 | ✅ | `PASS` / `FAIL` / `SKIP` / `ENV_BLOCKED` 之一 |
| 错误日志 | FAIL 时必须 | 完整 stderr/stdout，至少包含错误栈和关键告警 |
| 环境信息 | ✅（自动获取） | 执行 `scripts/collect-env.sh` 采集，无需用户提供 |

## 分析流程

### 步骤 1：输入校验

```
执行结果 == PASS？
  → 结论：用例已通过，无需诊断。终止。

执行结果 == SKIP？
  → 结论：用例被跳过。检查 skip 原因：
    - disabled_testcases 中 → 已知问题，确认是否需解除
    - 装饰器 skip → 分析 skip 条件在 NPU 下是否合理
    - 其他原因 → 记录

执行结果 == ENV_BLOCKED？
  → 结论：环境阻塞。明确阻塞原因（CANN 缺失/版本不匹配/编译产物缺失）
  → 建议：修复环境后重新运行

执行结果 == FAIL + 有日志？
  → 进入步骤 2
```

> ⚠️ 步骤 1 必做：检查 `disabled_testcases.json` 中是否已有该用例。若已存在则终止诊断，标记为已知问题。详见 `knowledge/regular/diagnosis-pitfalls.md` 陷阱 4。
>
> ⚠️ 步骤 1 必做：执行 `scripts/collect-env.sh` 采集环境信息，记录到报告的"基本信息"中。**必须实际执行，不得填写占位符。**

### 步骤 2：分布式检测与分流

判定用例是否为**分布式用例**。按以下双层规则检测：

**Tier 1 — 用例名模式匹配（优先，不需要日志）**：

命中任一模式即标记为分布式：

| 类别 | 模式 | 示例 |
|------|------|------|
| 多卡数量 | `_[2-9]\d*p`、`_multi_npu`、`_multi[_-]gpu` | `test_allreduce_2p` |
| HCCL 通信原语 | `_allreduce`、`_allgather`、`_broadcast`、`_reduce_scatter`、`_alltoall`、`_barrier`、`_send`、`_recv` | `test_broadcast_4p_npu` |
| 分布式框架 | `_distributed`、`_ddp`、`_fsdp`、`_process_group`、`_pipeline_parallel`、`_tensor_parallel` | `test_ddp_npu` |
| 文件/类路径 | 含 `distributed/`、`fsdp/`、`ddp/`；类名含 `Distributed`、`FSDP`、`DDP`、`ProcessGroup`、`MultiProcess` | `TestDistributedNNPU` |

**Tier 2 — 日志关键字匹配（用例名不明确时回退）**：

扫描 stderr 前 ~200 行，命中任一关键字即标记为分布式：

- `init_process_group`、`ProcessGroupHCCL`、`MASTER_ADDR=`、`WORLD_SIZE=`
- `HCCS LINK ERROR`、`HCCL OP RETRY FAILED`、`hcclCommAbort`
- `torchrun`、`watchdog timeout`、`flight recorder dump`

> Tier 1/Tier 2 的完整匹配表见 `knowledge/distributed/distributed-routing.md`。

---

**分流逻辑**：

```
if 分布式:
  → 执行步骤 1（分布式 Exit Code 处理）：见 knowledge/distributed/SKILL.md
  → 读取 knowledge/distributed/SKILL.md，按其步骤 1-5 执行分布式诊断
  → 分布式环境校验（RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT、NPU 设备数）记录到报告
else:
  → 读取 knowledge/regular/SKILL.md，按其步骤 2-5 执行常规诊断

两者完成后，回到本文件步骤 3 输出诊断报告。
```

> 两个子技能文件包含完整的步骤 2-5（错误特征提取 → 根因分类 → CANN 可行性判定 → 级联检测）。分布式子技能在部分步骤中引用常规子技能的基础逻辑（如 CANN 可行性三级判定、级联缺失检测），作为扩展而非重复。执行时只加载对应子技能，逐步骤完成分析后返回本文件。

### 步骤 3：输出报告

生成 `{workspace}/issues/{case_name}/根因分析.md`，按 `templates/diagnosis-report.md` 模板填充。

## 常见陷阱 (Gotchas)

- **不要只比较最终输出**：中间张量的差异才是根因线索，GPU 输出 float32 但 NPU 可能是 float16
- **不要跳过环境验证**：驱动版本不匹配会导致误导性错误
- **不要假设 fallback 告警就是根因**：同时出现 `aclrtMemcpy`/`copy_`/`Null_Pointer` 时，fallback 可能是 side effect
- **不要跳过 disabled_testcases 检查**：每次诊断必须先检查，避免对已知问题重复分析
- **连续三次相同诊断路径时停下来**：说明当前假设可能错误，重新审视原始错误信息
- **分布式场景不要只看单 rank**：HCCL 通信错误优先级高于算子错误，单 rank 超时可能是级联

## 何时必须问用户

只在以下情况才询问用户：

- 日志信息不足以判定（如缺少关键错误栈帧、没有 dtype 信息）
- 需要确认是否为已知问题（如是否在 disabled_testcases 中）
- 需要确认 CUDA 对照行为（用于区分上游问题）
- 日志中出现多个独立错误，需要确认分析哪个

环境信息通过 `scripts/collect-env.sh` 自动采集，不询问用户。分布式用例额外执行 `scripts/collect-env-distributed.sh`。

## 与完整流水线的关系

本技能是 `torch-npu-issue-pipeline-core` 中 **diagnose 阶段的只读子集**。覆盖步骤 1-4（前置检查 → 测试语义 → 错误解读 → 根因匹配），不覆盖修复实施、编译、验证、交付。

如果分析结论为"可修复"，建议用户使用 `torch-npu-issue-pipeline-core` 的完整流水线进行修复和交付。

## 参考资料

本技能的路由规则和判定逻辑基于以下本地知识文件。

### 常规诊断 (`knowledge/regular/`)

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| `routing.md` | 根因路由表、多特征冲突优先级、CANN 错误码速查、fallback 告警注意事项 | 步骤 2-3，错误码不在内联表中 / 多特征冲突 / fallback 可能是 side effect |
| `fixability-rules.md` | CANN 可行性三级判定流程、典型场景速查表 | 步骤 4，需要快速判定"可组合/不可组合"时对照速查表 |
| `missing-dispatch.md` | 算子未适配诊断策略：排除检查、CANN 终止规则、诊断搜索命令、适配层判断、代码路径参考 | 步骤 3（优先级 3）和步骤 4，特别是 fallback 同时出现 aclrtMemcpy/copy_/Null_Pointer 时 |
| `upstream-issue.md` | 上游问题判定步骤、CUDA 对照方法、调用栈分析、判定反例 | 步骤 3（优先级 2），区分"上游 bug"和"NPU 适配问题"时 |
| `diagnosis-pitfalls.md` | 6 类常见误判陷阱 | 步骤 1-3，特别是初判结论可能导致误判时对照检查 |
| `common-issues.md` | 7 类常见问题分类与快速参考（算子回退 CPU、接口不存在、类型检查失败、编译失败、Ran 0 tests、CUDA not available、Half 不支持） | 步骤 2，快速匹配已知问题模式 |

### 通用 (`knowledge/common/` — 常规和分布式共同引用)

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| `unsupported-patterns.md` | 6 类已知不支持模式泛化规则（dtype/维限/CUDA独有/格式限制/白名单算子/测试限制），含正则匹配和排除条件 | 步骤 2，常规和分布式均先跑此检查，命中直接终止避免重复分析 |

### 分布式诊断 (`knowledge/distributed/`)

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| `distributed-routing.md` | 分布式检测规则、扩展路由表（HCCL/Device Error）、HCCL 能力矩阵、Skip Code、多 rank 日志对比 | 步骤 2 分布式检测；步骤 2 多 rank 对比；步骤 2 优先级 1.5；步骤 4 HCCL 能力判定 |
| `distributed-pitfalls.md` | 6 类分布式特有误判陷阱 | 步骤 2-3，初判结论与分布式特征冲突时 |

> 步骤 2 中会先跑 `knowledge/common/unsupported-patterns.md` 的 6 类已知不支持快速判定。命中直接终止，不进入步骤 3-5 的完整分析流程。
