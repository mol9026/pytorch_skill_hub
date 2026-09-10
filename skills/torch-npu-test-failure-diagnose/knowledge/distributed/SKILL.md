---
name: torch-npu-test-distributed-diagnose
description: >-
  当 NPU 用例被判定为分布式用例时加载。执行步骤 1-5 的分布式诊断：
  Exit Code 处理、分布式错误特征提取、根因分类、HCCL 能力矩阵判定、级联缺失检测。
---

# 分布式用例诊断（步骤 1-5）

> 本文件由根 SKILL.md 在判定为分布式用例后加载。执行步骤 1-5 的分析，完成后返回根 SKILL.md 输出报告。

## 输入

从根 SKILL.md 继承以下输入：
- `case_name`、执行结果、错误日志、环境信息
- 分布式检测结果（匹配的 Tier 1/2 特征）

---

### 步骤 1：分布式 Exit Code 处理

```
执行结果 == SKIP (exit 75)?
  → 结论：多卡数量不足。记录用例所需卡数 vs 实际可用卡数
  → 建议：增加 NPU 卡数或降低用例卡数要求

执行结果 == SKIP (exit 76)?
  → 结论：HCCL 不可用。检查 libhccl.so 是否安装、HCCL 环境变量
  → 建议：安装/配置 HCCL 后重新运行

执行结果 == SKIP (exit 77)?
  → 结论：已知分布式限制。检查 disabled_testcases 对应条目
  → 建议：如已登记，终止诊断，标记为"已知分布式限制"
```

> 分布式环境校验信息（`RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`、NPU 设备数）需记录到报告的环境信息段。先执行 `scripts/collect-env.sh` 采集基础环境信息，再执行 `scripts/collect-env-distributed.sh` 采集分布式特有信息。

### 步骤 2：分布式错误特征提取

在常规诊断步骤 2 的 7 项基础提取之上（见 `knowledge/regular/SKILL.md` 步骤 2），补充提取以下特征：

| 提取项 | 搜索方式 | 对应根因类型 |
|--------|----------|-------------|
| HCCL 错误类型 | `HCCS LINK ERROR` / `UB LINK ERROR` / `HCCL OP RETRY FAILED` / `hcclCommAbort` | HCCL 通信故障 |
| 硬件设备错误 | `UCE ERROR` / `HBM.*ECC ERROR` / `SUSPECT MEM ERROR` / `SUSPECT REMOTE ERROR` | 硬件设备故障 |
| 超时/挂死信号 | `watchdog timeout` / `flight recorder dump` / `task timeout` / `error 107020` | 超时/挂死 |
| 多 rank 不一致 | `rank mismatch` / `broadcast to non-member` / `double init` | 多 rank 协调失败 |
| Tensor 分布错误 | `CPU tensor.*HCCL backend` / `mismatched dtypes across ranks` / `non-contiguous tensor` | 张量分布/布局问题 |
| HCCL 能力缺口 | `send/recv.*not supported` / `TORCH_NPU_USE_COMPATIBLE_IMPL` | HCCL 能力限制 |
| Rank 级联 | 各 rank 报不同错误 / 部分 rank 超时 | 多 rank 级联 |

**多 Rank 日志对比**（如有多 rank 日志）：

```
所有 rank 相同错误 → 单节点算子/环境问题（非通信特有）
部分 rank 错误，其他 rank 超时 → 错误 rank 是根因，超时 rank 是级联
各 rank 完全不同错误 → 多问题叠加，逐 rank 分析
```

> 完整的多 rank 日志对比方法和分析格式见 `knowledge/distributed/distributed-routing.md`。

**分布式相关代码搜索**（根据错误特征定位相关实现，具体搜索命令见 `knowledge/distributed/distributed-routing.md`）：

- HCCL 错误处理和通信实现 → `$TORCH_NPU_REPO/torch_npu/csrc/distributed/`
- HCCL 通信原语实现 → `$TORCH_NPU_REPO/torch_npu/csrc/distributed/HcclCompile.h`
- 分布式 patch / dispatch 注册 → `$TORCH_NPU_REPO/torch_npu/distributed/`
- HCCL 环境变量配置 → `$TORCH_NPU_REPO/torch_npu/csrc/core/npu/register/OptionsManager.h`

在进入分布式优先级路由之前，先将错误日志与已知不支持模式库进行快速判定。

**快速判定**：对照 `knowledge/common/unsupported-patterns.md` 中 6 类泛化规则，按类别 1→6 顺序匹配错误消息 + SampleInput 行。分布式用例同样会命中这些基础问题（dtype/维限/CUDA 独有/格式限制/白名单/测试限制）。

```
命中 → ⛔ cann-not-implemented → 终止诊断
全部不命中 → 进入步骤 3 分布式优先级路由
```

### 步骤 3：分布式根因分类

在常规优先级基础上，新增和扩展分布式特有规则。详细匹配条件和判定规则见 `knowledge/distributed/distributed-routing.md`。

#### 冲突优先级（分布式修正版）

```
优先级 1：环境/编译问题（含分布式环境）→ ✅ 可修复
优先级 1.5：HCCL/Device Error → ⛔ 不可修复（需硬件/HCCL 团队介入）
优先级 2：上游问题（含分布式扩展）→ ⚠️ 不可在 torch_npu 侧修复
优先级 3：算子未适配（含分布式扩展）→ 进入步骤 4
优先级 4：算子行为不一致（含分布式扩展）→ ✅ 可修复
优先级 5：内存问题（含多卡 HBM OOM）→ ⚠️ 有 workaround
优先级 6：超时/挂死 → 标记"需补充信息"
```

> **⚠️ 关键规则**：优先级 1.5（HCCL/Device Error）高于上游和算子问题。HCCL 通信通道断裂后，后续的算子错误是不可靠的 side effect。
>
> **补充参考**：
> - `knowledge/distributed/distributed-routing.md` — 完整分布式路由表、HCCL 能力矩阵、Skip Code 速查、多 rank 对比方法
> - `knowledge/distributed/distributed-pitfalls.md` — 6 类分布式特有误判陷阱

### 步骤 4：CANN 可行性判定（含 HCCL 能力矩阵）

当根因为"算子未适配"（优先级 3）时执行。基础三级判定见 `knowledge/regular/SKILL.md` 步骤 4，以下为分布式扩展：

#### HCCL 能力矩阵判定（通信原语特有）

当缺失算子涉及分布式通信原语时，增加第四级判定：

```
[分布式] 4. 查 HCCL 能力矩阵：
    ├── send / recv → HCCL 不支持 → ✅ 可修复（用 isend / irecv 替代）
    ├── recvAnysource → HCCL 不支持，无替代 → ⛔ cann-not-implemented
    ├── gather / scatter → 条件支持 → ✅ 条件可修复（设 TORCH_NPU_USE_COMPATIBLE_IMPL=1）
    └── 其他通信原语 → 查 knowledge/distributed/distributed-routing.md 中 HCCL 能力矩阵
```

> 完整 HCCL 能力矩阵（14 个操作的支持状态）见 `knowledge/distributed/distributed-routing.md`。

### 步骤 5：级联缺失检测（含多 rank 级联）

在常规级联检测（见 `knowledge/regular/SKILL.md` 步骤 5）基础上，增加：

**多 rank 级联检测**：

```
多个 rank 报不同错误时，记录 rank 级联关系：
  示例：rank 1 报 HCCS LINK ERROR（根因），rank 0/2 报 allreduce timeout（级联）
  报告中使用多 rank 级联表标记各 rank 的错误时间线和级联关系
```

> 具体搜索命令见 `knowledge/distributed/distributed-routing.md`。

---

> 分析完成。返回根 SKILL.md 的步骤 3 输出诊断报告。
