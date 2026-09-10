# 算子未适配 — 修复指导

> 本文件为诊断结论为"可修复"时提供修改建议和修复方向参考。**只读诊断技能可参考本文件在报告中提供修复伪代码/代码示例，但不得直接修改本地机器上的代码文件。** 修复代码由 `torch-npu-issue-pipeline-core` 的实施阶段执行。

## 强制路由规则

一旦根因判定为「NPU 后端缺少某算子/接口的原生实现」，修复 **必须**进入 NPU 侧正式接入开发流程（op-plugin / `npu_native_functions.yaml` / `torch_npu/csrc/aten/ops/` / aclnn / `torch.library` dispatch + NPU 原语合成），加载工具 `tools/npu-op-integration/SKILL.md`。

下列做法一律视为**反模式**：

1. 用 `torch.autograd.Function` 把 forward/backward 搬到 CPU 上算完再 `to(npu)`；
2. monkey-patch `torch.nn.functional.*` / `torch.Tensor.*` 在 NPU 分支里走 CPU；
3. 在 `torch_npu/utils/` 下新增"为绕过未适配算子而设计"的 CPU 兜底 patch；
4. **补完 forward 缺失的第一个 `aten::xxx` 后，看到 backward 还有 `aten::yyy / aten::zzz` 缺失就把 issue 返回为"根因已定位、待后续处理"**。这是"部分修复"，属于本策略反模式，必须沿调用链循环推进直至用例 PASS。

## CPU Fallback 场景的修复优先级

当错误特征为「算子走 CPU fallback」（告警 `will fall back to run on the CPU`）时，**必须按以下优先级选择修复路径**：

| 优先级 | 修复路径 | 适用条件 | 示例 |
|--------|---------|---------|------|
| **1（首选）** | **在 NPU 上注册该算子的原生实现** | CANN/aclnn 已有对应能力，或可通过已有 NPU 算子组合实现 | `quantize_per_tensor_dynamic` 可由已有的 `quantize_per_tensor` + 动态范围计算组合 → 注册 NPU kernel |
| **2（次选）** | 修复 fallback 回传基础设施 | 算子本身实现复杂度高或 CANN 无对应能力，且 fallback 回传路径断裂（如 `.to(npu)` 在量化张量上失败） | `empty_quantized` + `_copy_from` 缺少 `QuantizedPrivateUse1` 注册 → 补注册 |

**决策流程**：

```
发现算子 aten::xxx 走 CPU fallback
│
├── 0. 首先：查看 CUDA 后端如何实现 aten::xxx
│   ├── 搜索 PyTorch 源码中 aten::xxx 的 CUDA dispatch 实现
│   ├── 理解 CUDA 实现的核心逻辑
│   └── 以 CUDA 实现为参考，评估 NPU 侧的对标实现方案
│
├── 1. 关键判定：CUDA 是否有原生实现（不走 CPU fallback）？
│   ├── 【是：CUDA 有原生实现】→ ⛔ 硬规则：NPU 也必须注册原生实现
│   │   ├── CUDA 实现是纯路由/组合逻辑 → NPU 直接用相同逻辑（模式 A/C2）
│   │   ├── CUDA 调用 cuBLAS/cuDNN → 检查 CANN/aclnn 是否有对标接口
│   │   │   ├── 有 → 注册 NPU kernel（模式 B/C/C2）
│   │   │   └── 无 → 检查是否可用已有 NPU 算子组合实现
│   │   │       ├── 可组合 → 用已有 NPU 算子组合实现（模式 A/C2）
│   │   │       └── 不可组合 → ⛔ **CANN 算子未实现，终止修复**
│   │   └── 以上均不可行 → ⛔ **CANN 算子未实现，终止修复**
│   │
│   └── 【否：CUDA 也走 fallback 或无 CUDA 实现】→ NPU 走 fallback 是合理对齐
│       ├── 检查 fallback 回传路径是否完整
│       │   ├── 回传路径断裂 → 修复回传基础设施（优先级 2）
│       │   └── 回传路径正常 → 记录为「已知 fallback，与 CUDA 行为一致」
│       └── 如果有性能需求，可选择性注册 NPU 原生实现（优化项，非必须）
│
└── 2. 无论走哪条路径，在根因分析.md 中必须记录：
    - CUDA 实现方式概述（1-3 句话）
    - CUDA 是否走 fallback（是/否）
    - NPU 选择的修复路径及理由
    - fallback 的性能影响评估（如适用）
```

> **硬规则：CUDA 有原生实现 → NPU 禁止走 CPU fallback**
>
> 这是本策略的最高优先级判定。理由：
> - CUDA 有原生实现意味着该算子是生产级 GPU 必备能力，NPU 作为对标后端不能降级；
> - 走 CPU fallback 会引入 H2D/D2H 搬运开销，且**掩盖算子缺口**——所有调用方都静默走 CPU；
> - 注册 NPU kernel 从根源消除 fallback，后续所有调用路径都受益。

## 修复模式

| 模式 | 适用 | 需编译 |
|------|------|--------|
| A：Python 适配层 | 已有 NPU 算子组合 / 行为选路 | 否 |
| B：yaml 注册 | op-plugin 已有 C++ 实现，未声明 dispatch | 是 |
| C：op-plugin 新增 | 需调用 CANN aclnn 接口 | 是 |
| C2：torch.library.impl | PyTorch 官方 dispatch，标准算子后端注册 | 否 |

**选择决策树**：

```
根因是否为「NPU 缺少该算子的原生实现」？
├── 是 → 走 NPU 接入流程（tools/npu-op-integration）
│        ├── 已有 op-plugin C++ 实现未注册 → 模式 B
│        ├── 需调用 CANN aclnn 接口 → 模式 C
│        └── 只需补 dispatch 路由 → 模式 C2
└── 否（Python 包装 / 路由缺失）
         ├── 有标准 aten 签名 → 模式 C2
         └── 无标准 aten 签名 → 模式 A（仅限行为选路，禁止绕缺失 kernel）
```

### 模式选择红线（Hard Rules）

以下判定一旦命中，**禁止**使用模式 A（Python monkey-patch），必须改走模式 B / C / C2：

| 判定 | 含义 | 禁止的做法 |
|---|---|---|
| 错误日志出现 `The operator 'aten::<op>' is not currently supported on the NPU backend` | NPU 确实没有 kernel | 用 `autograd.Function` 在 NPU 路径上把 forward/backward 搬到 CPU |
| 错误日志出现 `will fall back to run on the CPU` | NPU 正在走 CPU fallback | 不作为就通过测试；应补 NPU 实现 |
| 被替换的是 **生产 API**（如 `F.linear`、`Tensor.to_dense`、`Tensor.matmul`） | 会污染所有 NPU 用户调用 | 绝不允许在 `torch_npu/utils/` 里全局替换这类顶层 API 以绕过算子缺失 |
| 需要反向传播正确性（用例包含 `autograd.grad` / `loss.backward()`） | CPU 兜底 + 数据搬运会丢语义 / 丢 dtype / 丢 layout | 不得在 Python 层拼 backward；必须让 NPU 上的算子分发自身正确 |

### 级联缺失算子处理

一个用例经常**不是只缺一个算子**——forward 补完 `aten::addmm` 后，backward 会拉出 `aten::matmul_backward`；`to_dense` 的反向又拉出 `aten::sparse_mask` → `aten::mul.out`；autograd 展开还可能引入 layout 不一致的问题。它们**只有在前一个算子修好后**才暴露。

**正确处理流程**：

1. 每轮只修一个 `aten::xxx` 的缺失（本策略 + `tools/npu-op-integration` 单算子约束）；
2. 修好后**立刻 re-run 原用例**，观察是否还有新的 `not currently supported` / `no kernel found`；
3. 有 → 回到步骤 1，以新 `aten::yyy` 为对象重新走一遍本策略（不关闭 issue、不写"待后续"结论）；
4. 无 → 进入 verify。

**调度器**：当一个用例需要循环 ≥ 2 次时，由 `tools/missing-dispatch-loop/SKILL.md` 负责维护级联表、循环条件和退出条件；本策略负责每一轮内部的"识别 + 适配"。

**禁止在级联场景做的事**：

- ❌ 把中间某一轮的用例加进 `disabled_testcases.json`，说"分步修复，这一步先跳过"；
- ❌ 补完 forward 就写"PASS"：forward-only 通过不是 PASS，完整用例退出码 0 才算 PASS；
- ❌ 把某一轮的缺失归给"上游问题"——真正的上游问题应该在 CUDA 上也失败，本策略覆盖的都是 NPU 独有的缺失。

**循环退出标准**（与 `tools/npu-op-integration` 阶段 6 一致）：

1. `python -W error::UserWarning -m pytest <原用例>` 退出码 0；
2. 近邻参数化用例（不同 dtype / layout / shape / bias 组合）均无新增失败；
3. `grep -nE "\.cpu\(\)|to\(['\"]cpu['\"]\)" <本次新增文件>` 为空。

### 反面案例：sparse Linear / to_dense 的 CPU 兜底 monkey-patch（禁止效仿）

- **用例**：`TestNN.test_linear_autograd_device_cuda_{bias,nobias}_weightCOO`
- **触发现象**：
  - forward：`F.linear(input_npu, sparse_coo_weight_npu, bias)` → `aten::addmm not supported on NPU`
  - backward：`expected = einsum(..., weight.to_dense()) + bias` 反向 → `aten::mul.out not supported on NPU`（来自 `to_dense` 的 `sparse_mask`）
- **错误的"修复"做法（已执行过的反例，不要重复）**：
  1. 新增 `torch_npu/utils/_sparse_patches.py`，用两个 `torch.autograd.Function`（`_NpuSparseLinear` / `_NpuSparseToDense`）在 sparse 路径上把整个 forward + backward 搬到 CPU；
  2. 在 `torch_npu/utils/__init__.py` / `torch_npu/__init__.py::_apply_class_patches()` 中全局替换 `torch.nn.functional.linear` 和 `torch.Tensor.to_dense`；
  3. 通过把 NPU 张量 `.cpu()` 再 `.to(npu)` 来"回避" `aten::addmm(sparse)` / `aten::mul.out(sparse)` 在 NPU 上的缺失。
- **为什么这是反模式**：
  - **污染生产 API**：`F.linear` 和 `Tensor.to_dense` 被全局替换，任何 NPU 用户的 sparse 调用都被迫走 CPU，性能损失静默发生，且与"NPU 张量留在 NPU"这一基本语义相冲突；
  - **掩盖真实缺口**：NPU 后端实际缺的是 sparse 输入版本的 `aten::addmm` 和 `aten::mul.out`，CPU 兜底让问题看起来"修好了"但算子空缺依然存在，后续其它用例仍会踩坑；
  - **无法通过审查**：这类改动在 `torch_npu` 仓库 code review 阶段会被要求回退；
  - **反向语义脆弱**：手工拼 `torch.autograd.grad` + sparse `_indices()/_values()` 重建极易在 dtype / layout / coalesce 语义上出错，只是在 4x4 小用例上碰巧通过。
- **正确路径**：把该用例归档为「算子接入类问题」，加载 `tools/npu-op-integration/SKILL.md`，沿 NPU 侧正式开发路径持续推进直至真实接入完成。

### 实战案例：aten::_fused_sdp_choice NPU 适配

> ⚠️ **语义陷阱警告**：当算子注册涉及 backend 选择（如 `_fused_sdp_choice`）时，必须先确认测试类语义（CudaOnly vs XpuOnly）。NPU 作为 PrivateUse1 后端应返回 `OVERRIDEABLE`(4) 而非模仿 CUDA 返回 `FLASH_ATTENTION`(1) / `EFFICIENT_ATTENTION`(2)。

| 项目 | 内容 |
|------|------|
| **问题** | 该算子在 NPU 上无 dispatch，fallback 到 CPU，导致 backend 选择逻辑与 NPU 实际能力不符 |
| **错误表现** | `AssertionError: 1 != 2`（期望 `EFFICIENT_ATTENTION`，实际返回 `FLASH_ATTENTION`） |
| **修复方式** | 使用 `torch.library.impl` 注册 PrivateUse1 后端实现 |
| **修复文件** | `torch_npu/utils/_sdp_utils.py`（新增）、`torch_npu/utils/__init__.py`（导入）、`torch_npu/__init__.py`（导入） |

该案例说明：即使算子 fallback 到 CPU 后数值计算无误，返回值仍可能因为硬件感知逻辑（如 `_fused_sdp_choice` 根据设备能力选择 backend）而导致断言失败。此时不能依赖 fallback，必须为 NPU 注册正确的 dispatch 实现。

## 验证修复

修复后运行：

```bash
# 验证算子不再 fallback，结果在 NPU 上
python -c "
import torch
import torch_npu

t = torch.randn(4, 4).npu()
result = torch.op_name(t)
print('device:', result.device)   # 应为 npu:0
print('no fallback')
"

# 运行原失败测试
python -m pytest <测试文件> -k <测试名> -v -W error::UserWarning
```

`-W error::UserWarning` 会把 fallback 告警变成错误，确认告警已消除。

## 注意事项

1. **fallback 不等于测试通过**：算子 fallback 到 CPU 后，数值可能是对的，但设备不对，断言仍会失败。
2. **区分 aten 算子和 torch 函数**：`aten::op` 的 NPU 注册和 `torch.op` 的 Python 包装是两个层次，都可能需要适配。
3. **op-plugin 修改需重新编译**：修改 `third_party/op-plugin/` 后必须重新编译 torch_npu。
4. **CANN 版本兼容性**：新加的 aclnn 算子调用要用 `DO_COMPATIBILITY` 包裹，确保在旧版 CANN 上能回退。
5. **注意 in-place 和 out 变体**：`aten::op_`（in-place）和 `aten::op.out`（out 参数）通常需要独立注册。
