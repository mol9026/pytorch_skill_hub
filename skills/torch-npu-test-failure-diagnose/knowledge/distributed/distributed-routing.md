# 分布式根因路由规则

> 用于诊断阶段：当用例被检测为分布式用例时，补充单卡路由规则。不替代 `routing.md`，而是作为其扩展。

## 分布式检测规则

### Tier 1：用例名模式匹配（优先，不需要日志）

| 类别 | 模式 / 正则 | 示例 |
|------|------------|------|
| 多卡数量 | `_[2-9]\d*p`、`_multi_npu`、`_multi[_-]gpu` | `test_allreduce_2p`、`test_fsdp_4p` |
| HCCL 通信原语 | `_allreduce`、`_allgather`、`_broadcast`、`_reduce_scatter`、`_alltoall`、`_barrier`、`_send`、`_recv`、`_irecv`、`_isend` | `test_broadcast_4p_npu` |
| 框架关键词 | `_distributed`、`_ddp`、`_fsdp`、`_deepspeed`、`_megatron`、`_process_group`、`_pipeline_parallel`、`_tensor_parallel`、`_sequence_parallel`、`_zero[_-]` | `test_ddp_npu`、`TestFSDP` |
| 文件/类路径 | 文件路径含 `distributed/`、`fsdp/`、`ddp/`；类名含 `MultiProcess`、`Distributed`、`FSDP`、`DDP`、`ProcessGroup` | `TestDistributedNNPU` |

### Tier 2：日志关键字匹配（用例名不明确时回退）

扫描 stderr 前 ~200 行，命中以下任意关键字即判定为分布式：

| 关键字 | 置信度 | 含义 |
|--------|--------|------|
| `init_process_group` / `dist.init_process_group` | 高 | 进程组初始化 |
| `MASTER_ADDR=` / `MASTER_PORT=` / `WORLD_SIZE=` | 高 | 分布式环境变量 |
| `ProcessGroupHCCL` | 高 | HCCL 后端栈帧 |
| `hcclCommAbort` / `hcclCommInitRank` / `HcclAllReduce` | 高 | HCCL 通信器调用 |
| `torchrun` / `torch.distributed.run` / `torch.distributed.launch` | 高 | 分布式启动器 |
| `HCCS LINK ERROR` / `UB LINK ERROR` / `HCCL OP RETRY FAILED` | 高 | HCCL 通信故障 |
| `UCE ERROR` / `HBM MULTI BIT ECC ERROR` / `SUSPECT MEM ERROR` / `SUSPECT REMOTE ERROR` | 高 | 硬件设备故障 |
| `watchdog timeout` / `heartbeat monitor` / `flight recorder dump` | 高 | 超时/挂死 |
| `rank mismatch` / `broadcast to non-member` | 高 | 多 rank 协调失败 |
| `DistributedDataParallel` / `FSDP` / `DeepSpeed` | 中 | 分布式训练框架 |

---

## 扩展路由表

以下优先级在单卡路由基础上扩展。单卡的 6 个优先级保留，新增优先级 1.5。

### 冲突优先级（分布式修正版）

1. **环境/编译问题优先**（含分布式环境）
2. **HCCL/Device Error**（新增 — 高于上游和算子）
3. **上游问题**
4. **算子未适配 > 算子行为不一致**
5. **内存问题独立处理**
6. **级联缺失识别**（含多 rank 级联）

---

### 优先级 1：环境/编译问题（分布式扩展）

在单卡匹配条件基础上，新增：

| 分布式匹配条件 | 说明 |
|---------------|------|
| `torch_npu not compiled with HCCL support` | HCCL 编译时未启用 |
| `libhccl.so: cannot open shared object file` | HCCL 动态库缺失 |
| `init_process_group` failed: `MASTER_ADDR` not set | 分布式环境变量缺失 |
| `init_process_group` failed: `MASTER_PORT` not set | 同上 |
| `init_process_group` failed: `WORLD_SIZE` / `RANK` not set | 同上 |
| `trying to initialize the default process group twice` | 重复初始化 |
| `Address already in use` (MASTER_PORT) | 端口冲突 |
| `torch.distributed` not available | PyTorch 分布式模块不可用 |

**根因标注**：环境/编译问题（分布式）
**可修复性**：✅ 可修复（配置环境变量 / 安装 HCCL / 换端口）

---

### 优先级 1.5：HCCL/Device Error（新增 — 仅分布式）

> ⚠️ 此优先级高于上游问题和算子问题。HCCL 通信通道断裂后，后续的算子错误是不可靠的 side effect。

#### 1.5a — HCCL 通信错误

| 匹配条件 | 详细分类 |
|---------|---------|
| `HCCS LINK ERROR` | 芯片间互联（HCCS）链路故障 |
| `UB LINK ERROR` | Unified Buffer 链路故障 |
| `HCCL OP RETRY FAILED` | HCCL 操作重试耗尽 |
| `hcclCommAbort` | HCCL 通信器异常终止 |
| `HCCL error code`（任意非零） | HCCL 层报错 |
| `Device link error` | 设备间链路故障 |
| 飞行记录器 dump 中出现上述错误 | 超时后 dump 出的 HCCL 错误 |

**根因标注**：HCCL 通信错误
**可修复性**：⛔ 不可修复（需硬件/HCCL 团队介入）
**处理**：终止诊断，收集 flight recorder dump + 所有 rank 的 HCCL 日志，联系 HCCL/硬件团队

#### 1.5b — 硬件设备错误

| 匹配条件 | 详细分类 |
|---------|---------|
| `UCE ERROR` | 不可纠正硬件错误（Uncorrectable Error） |
| `HBM MULTI BIT ECC ERROR` | HBM 多比特 ECC 错误 |
| `SUSPECT MEM ERROR` | 疑似内存硬件错误 |
| `SUSPECT REMOTE ERROR` | 疑似远端设备硬件错误 |
| `NPU hardware fault` / `device reset` | NPU 硬件故障/复位 |

**根因标注**：硬件设备错误
**可修复性**：⛔ 不可修复（需硬件团队介入）
**处理**：终止诊断，收集设备健康检查日志，联系硬件团队

---

### 优先级 2：上游问题（分布式扩展）

| 分布式匹配条件 | 说明 |
|---------------|------|
| 错误栈无 `torch_npu` 帧且无 `ProcessGroupHCCL` 帧 | 纯上游分布式模块 bug |
| 测试代码硬编码 `backend='nccl'`（应适配为 `backend='hccl'`） | 测试适配问题 |
| PyTorch 原生 `torch.distributed` 模块 bug | 如 TCPStore 超时（非 HCCL 特有） |
| CUDA 分布式同样失败 | NPU 非独有 |

**根因标注**：上游问题（分布式）
**可修复性**：⚠️ 不可在 torch_npu 侧修复

---

### 优先级 3：算子未适配（分布式扩展）

| 分布式匹配条件 | 说明 |
|---------------|------|
| 单 rank 内出现 fallback 告警（与单卡场景相同） | 确认其他 rank 是否同样出现 |
| `gather` / `scatter` 在不设置 `TORCH_NPU_USE_COMPATIBLE_IMPL=1` 时失败 | 缺少兼容模式配置 |
| `send` / `recv` 操作失败 | HCCL 不支持同步 send/recv |
| `recvAnysource` 操作失败 | HCCL 不支持，无替代方案 |
| `alltoall` / `alltoall_single` 失败 | 检查 dtype / shape 约束 |

**根因标注**：算子未适配（分布式通信原语）
**可修复性**：→ 进入步骤 3，走 HCCL 能力矩阵判定

---

### 优先级 4：算子行为不一致（分布式扩展）

| 分布式匹配条件 | 说明 |
|---------------|------|
| 各 rank 上 dtype 不一致导致精度差异 | 跨 rank 数据分布问题 |
| 非 contiguous tensor 输入 collective 操作 | HCCL 要求 contiguous |
| allreduce 等操作在不同 rank 产生不一致的结果 | 集合操作正确性问题 |
| `assert_close` 失败且差异与 rank 相关 | rank 相关精度差异 |

---

### 优先级 5：内存问题（分布式扩展）

| 分布式匹配条件 | 说明 |
|---------------|------|
| 部分 rank OOM，其他 rank 正常 | 多卡内存分布不均衡 |
| 所有 rank 同时 OOM | 模型/数据总内存超限 |
| HCCL workspace 内存过大 | 通信缓冲区内存 |

---

### 优先级 6：超时/挂死（分布式扩展）

当无法归类到以上优先级，且日志仅含超时信号时：

| 匹配条件 | 说明 |
|---------|------|
| `watchdog timeout` 但无可读错误 | 不知道哪个 rank/操作触发了超时 |
| `flight recorder dump` 但无可读错误 | 飞行记录器未捕获到明确错误 |
| `task timeout` / `error 107020` | CANN 超时但无进一步信息 |

**根因标注**：超时/挂死（需补充信息）
**处理**：标记为"需补充信息"，列出所需：所有 rank 日志、flight recorder dump、网络诊断、HCCL 环境变量配置

---

## HCCL 能力矩阵速查

| 操作 | 支持状态 | 条件 / 替代方案 |
|------|---------|---------------|
| `allreduce` | ✅ 完全支持 | |
| `broadcast` | ✅ 完全支持 | |
| `allgather` | ✅ 完全支持 | |
| `allgather_into_tensor` | ✅ 完全支持 | |
| `reduce_scatter` | ✅ 完全支持 | |
| `reduce_scatter_tensor` | ✅ 完全支持 | |
| `alltoall` | ✅ 完全支持 | |
| `alltoall_single` | ✅ 完全支持 | |
| `barrier` | ✅ 完全支持 | |
| `allgather_coalesced` | ✅ 完全支持 | |
| `all_reduce_coalesced` | ✅ 完全支持 | |
| `gather` | ⚠️ 条件支持 | 需 `TORCH_NPU_USE_COMPATIBLE_IMPL=1`，否则降级为 allgather |
| `scatter` | ⚠️ 条件支持 | 需 `TORCH_NPU_USE_COMPATIBLE_IMPL=1`，否则降级为 broadcast |
| `send` | ❌ 不支持 | 用 `isend` 替代 |
| `recv` | ❌ 不支持 | 用 `irecv` 替代 |
| `recvAnysource` | ❌ 不支持 | 无替代方案 |

---

## 分布式 Skip Code 速查

| Exit Code | 含义 | 处理建议 |
|-----------|------|---------|
| 75 | 多 NPU 卡数不足 | 增加卡数或降低用例卡数要求 |
| 76 | HCCL 不可用 | 安装/配置 HCCL 库 |
| 77 | 已知分布式问题 | 检查 disabled_testcases 中的对应条目 |

---

## 多 Rank 日志对比分析方法

当有多个 rank 的日志时，按以下维度对比：

```
1. 错误一致性
   ├── 所有 rank 相同错误 → 单节点算子/环境问题（非通信特有）
   ├── 部分 rank 错误，其他 rank 超时 → 错误 rank 是根因，超时 rank 是级联
   └── 各 rank 完全不同错误 → 多问题叠加，逐 rank 分析

2. 时间线
   ├── 同一时间戳 → 同步触发（环境/编译问题可能性大）
   └── 不同时间戳 → 级联触发（根因 rank 先报错，其他 rank 后超时）

3. 输入数据
   ├── 各 rank 输入相同 → 排除数据分布问题
   └── 各 rank 输入不同 → 可能数据相关（特定输入触发边界条件）
```

### 对比输出格式

```
多 Rank 日志对比:
  | Rank | 错误类型 | 关键错误信息 | 时间 |
  |------|---------|-------------|------|
  | 0    | timeout | allreduce timeout | T+5s |
  | 1    | HCCL    | HCCS LINK ERROR  | T+0s |
  | 2    | timeout | allreduce timeout | T+5s |

  分析：Rank 1 在 T+0s 报 HCCS LINK ERROR（根因），Rank 0/2 在 T+5s 超时（级联）。
  结论：根因在 Rank 1 的网络链路，非算子问题。
```

---

## 错误码参考

| 错误码 | 含义 |
|--------|------|
| 107020 | CANN task timeout（分布式常见） |
| 200 | ErrCode::HCCL（通用 HCCL 错误） |
| `HCCL_OP_RETRY_FAILED` | HCCL 操作重试失败（队列状态 10） |

## 相关环境变量

| 变量 | 作用 |
|------|------|
| `TORCH_NPU_USE_COMPATIBLE_IMPL=1` | 启用 gather/scatter 兼容实现 |
| `TORCH_HCCL_ENABLE_MONITORING` | 启用心跳监控线程 |
| `TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC` | 心跳超时阈值 |
| `TORCH_HCCL_DUMP_ON_TIMEOUT` | 超时时触发 flight recorder dump |
| `TORCH_HCCL_EXEC_TIMEOUT` | HCCL 操作执行超时 |
| `HCCL_CONNECT_TIMEOUT` | HCCL 连接超时 |
| `HCCL_EVENT_TIMEOUT` | HCCL 事件超时 |
