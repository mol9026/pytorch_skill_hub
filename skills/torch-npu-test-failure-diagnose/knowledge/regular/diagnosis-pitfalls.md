## 常见误判与陷阱

### 陷阱 1：混淡"测试语义不适配"和"算子行为不一致"

**症状**：断言失败，预期值与实际值不同
**误判**：直接判定为"算子结果错误"，开始分析精度问题
**正确流程**：先执行第 1 步"测试类语义适用性检查"。如来自 `*CudaOnly` 类且无对应 `*XpuOnly` 版本，应标记为"测试适配缺失"而非"算子结果错误"。

### 陷阱 2：误认为"已有 patch 就不用诊断"

**症状**：在 torch_npu 子模块中找到了目标函数的 NPU patch
**误判**：认为 patch 已存在，问题不在适配层
**正确流程**：检查 patch 是否在当前测试执行路径上**实际生效**。如 patch 位于惰性导入的子模块中（如 `torch_npu._inductor`），测试可能在 patch 导入前就已失败。需确认 patch 的导入时序。

### 陷阱 3：修复后仓促结束，遗漏新阻塞点

**症状**：第一阶段修复成功，但初步验证遇到新的不同错误
**误判**：在同一报告中试图一次性修复所有问题
**正确流程**：如新错误与当前根因无直接关系（如修了设备限制后遇到缺少依赖包），标注"第一阶段根因已修复，新问题待后续处理"。一次处理一个根因。

### 陷阱 4：未利用 disabled_testcases.json 导致重复工作

**症状**：投入时间诊断后发现是已知问题
**误判**：跳过第 0 步前置检查直接开始分析
**正确流程**：每次诊断必须执行第 0 步，`grep <用例名> disabled_testcases.json`。如找到条目，停止诊断并询问用户。

### 陷阱 5：用 Python 层 CPU 兜底 monkey-patch 绕过缺失算子（禁止）

**症状**：错误日志中明确看到 `operator 'aten::xxx' is not currently supported on the NPU backend` / `will fall back to run on the CPU`，即 NPU 后端**本身没有**该算子/接口的实现。
**误判**：在 `torch_npu/utils/` 下新增 `torch.autograd.Function` 或 monkey-patch，把 `F.xxx` / `Tensor.xxx` 在 NPU 路径上搬到 CPU 计算再搬回来，用来"让测试过"。
**为什么不对**：
- 这类 patch 会**全局污染**生产 API（所有 NPU 用户都被迫走 CPU），且掩盖了真正的 kernel 空缺；
- 反向传播时手工拼 `torch.autograd.grad` + sparse `_indices()/_values()` 极易丢 layout / dtype / coalesce 语义；
- 无法通过 torch_npu 仓库的 code review，属于确定会被要求回退的改法。

**正确流程**：
1. 将根因标注为**算子接入类问题**；
2. 在报告中明确标注"需要 NPU 侧正式接入开发"，修复方向为 op-plugin（aclnn）→ `npu_native_functions.yaml` → `torch_npu/csrc/aten/ops/` 的路径，具体修复指导见 `references/missing-dispatch-guide.md`；
3. 若外部依赖（CANN aclnn、op-plugin 排期）暂未就绪，在报告中标注"阻塞中（继续推进）"，**不得**以"短期无法排期"为由建议新增 `disabled_testcases.json` 条目结束 issue，更不得建议用 CPU 兜底 patch 遮盖。

**具体反例**：`TestNN.test_linear_autograd_device_cuda_{bias,nobias}_weightCOO`（sparse COO weight 的 `F.linear` / `Tensor.to_dense` CPU 兜底 monkey-patch）——详见 [references/missing-dispatch-guide.md 的「反面案例」一节](../../references/missing-dispatch-guide.md#反面案例sparse-linear--to_dense-的-cpu-兜底-monkey-patch禁止效仿)。

### 陷阱 6：级联缺失算子时以"部分修复"名义结束 issue（禁止）

**症状**：第 1 轮补上了 forward 的 `aten::addmm`，re-run 用例后又报 `aten::matmul_backward not currently supported` / `aten::sparse_mask not currently supported` / `aten::mul.out not currently supported`。
**误判**：写"forward 根因已修复，backward 依赖算子未适配、等待后续接入"，把 issue 标为"已定位、待后续处理"或"已修复一部分"结束。
**为什么不对**：
- 这种级联缺失是**同一条调用链上的同一个根因**（算子未适配）在连续轮次中暴露的多个点，不是多个独立根因，更不是"根因已定位"的充分条件；
- `missing-dispatch` 策略已经明确要求沿调用链循环（见策略文件的「级联缺失算子处理」节），止步于 forward 就相当于默认替代了本应由 `missing-dispatch-loop` 做的调度；
- forward 通过 ≠ 用例 PASS；用例退出码 0 才算本阶段的唯一通过标准。

**正确流程**：
1. 在报告中标注"级联缺失"，列出所有已识别的缺失算子链；
2. 在修复方向中标注"需要沿调用链循环修复，每轮补一个算子后 re-run"，具体级联处理指导见 `references/missing-dispatch-guide.md`；
3. 在报告中保留每一轮暴露的算子名，直至完整调用链上所有算子均已识别；
4. forward 通过 ≠ 用例 PASS；用例退出码 0 才算最终通过标准。

**具体反例**：同上 sparse linear 用例——若在第 1 轮补完 `aten::addmm` 就返回"根因已定位"结束，后续 `aten::matmul_backward` / `aten::sparse_mask` / `aten::linear` autograd 三轮都会被当作"下次再说"而永远不做；正确做法是循环 4 轮直到用例真实 PASS。
