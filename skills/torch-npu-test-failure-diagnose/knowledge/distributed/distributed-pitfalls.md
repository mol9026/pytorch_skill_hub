## 分布式诊断常见误判与陷阱

### 陷阱 1：单 rank 日志分析误判全局

**症状**：只有 rank 0 的日志（最常见的场景），其他 rank 日志缺失。
**误判**：根据 rank 0 的错误信息直接判定根因。例如 rank 0 报 `allreduce timeout`，直接判为"HCCL 通信超时"。
**正确流程**：
- 如果 rank 0 报 `allreduce timeout` 而 rank 1-3 日志缺失，根因很可能在 rank 1-3（网络故障 / OOM / 崩溃），rank 0 只是等待超时
- 如果多 rank 日志可以获得，必须对比各 rank 的错误后下结论
- 如果只有单 rank 日志，在报告中明确标注"仅 rank N 日志可用，其他 rank 状态未知"

### 陷阱 2：CCL 通信错误与算子错误混淆

**症状**：日志中同时出现 HCCL 错误（如 `HCCS LINK ERROR`）和算子错误（如 `aclnnAdd failed`）。
**误判**：认为是算子缺失导致级联失败，按"算子未适配"处理。
**正确流程**：
- HCCL 通信错误应**优先处理**（优先级 1.5 > 优先级 3）
- 通信通道断裂后，后续的算子错误是不可靠的 side effect
- 先排查根因 HCCL 错误（链路故障、重试耗尽等），算子错误在通信恢复后可能自然消失
- 在报告中明确标注"算子错误为 HCCL 通信故障的副作用，非独立根因"

### 陷阱 3：忽略 rank 间一致性

**症状**：rank 0 PASS 但 rank 1 FAIL，或各 rank 报不同错误。
**误判**：判定为 rank 1 的算子问题，或各 rank 独立分析。
**正确流程**：
- 单 rank 失败往往是数据分布不均匀或 race condition
- 对比各 rank 的输入数据、dtype、tensor 形状
- 如果各 rank 输入相同但行为不同 → 可能是硬件差异或环境差异
- 如果各 rank 输入不同但行为不同 → 可能是特定输入触发边界条件

### 陷阱 4：忽略 nccl → hccl 后端替换

**症状**：测试代码中硬编码 `backend='nccl'`，日志报 `NCCL not available` 或 fallback 到 CPU。
**误判**：标记为 ENV_BLOCKED（NCCL 不可用），或去安装 NCCL。
**正确流程**：
- NPU 上的分布式后端是 HCCL，不是 NCCL
- 检查测试代码是否有 nccl→hccl 的自动替换（`torch_npu.contrib.transfer_to_npu`）
- 如果没有自动替换或替换不生效 → 根因为测试适配问题（优先级 2，上游/测试侧）
- 正确方向：将 `backend='nccl'` 适配为 `backend='hccl'`，或在测试中使用 `dist.Backend.HCCL`
- 如果测试中有 `dist.is_nccl_available()` 守卫 → 需替换为 `dist.is_hccl_available()`

### 陷阱 5：忽略 subprocess 执行模式

**症状**：错误日志来自 stderr 字符串匹配，而非 Python exception traceback。
**误判**：用常规的 `Traceback (most recent call last)` 方式提取错误栈。
**正确流程**：
- `torchrun` 启动的分布式用例错误通过 stderr 输出，不是 Python exception
- 需要关注：subprocess crash、signal（SIGKILL/SIGTERM/SIGABRT）、C++ assert 输出
- `RuntimeError` 可能以纯文本而非 traceback 格式出现
- 超时场景：subprocess 可能被外部 watch dog 杀死，日志末尾为 `[ERROR] Killed by signal`
- 适应点：搜索 `RuntimeError`、`[ERROR]`、`Traceback`、`signal`、`SIGKILL`、`SIGABRT` 等多模式

### 陷阱 6：忽略 HCCL 条件能力

**症状**：gather/scatter 操作失败，或 send/recv 操作失败。
**误判**：gather/scatter → 标记为 HCCL 不支持（错误，其实条件支持）；send/recv → 试图找替代的 CANN API。
**正确流程**：

| 操作 | 正确判定 | 修复方向 |
|------|---------|---------|
| `gather` 失败 | 未设 `TORCH_NPU_USE_COMPATIBLE_IMPL=1` → 环境配置问题（可修复） | 设置环境变量 |
| `scatter` 失败 | 同上 | 设置环境变量 |
| `send` / `recv` 失败 | HCCL 不支持同步 send/recv → 算子未适配（可修复） | 用 `isend` / `irecv` 替代 |
| `recvAnysource` 失败 | HCCL 不支持，无替代方案 → cann-not-implemented（不可修复） | 向 CANN 团队提交需求 |

- 先查 HCCL 能力矩阵再判定，不要根据直觉
- 条件支持的操作先检查环境变量是否已设置
