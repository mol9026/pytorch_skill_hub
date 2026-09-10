# 已知不支持模式速查

> 用于步骤 2：在进入 6 级优先级路由之前，先检查错误日志是否命中已知不支持的场景。命中即终止诊断，标记为"已知不支持（cann-not-implemented）"，避免对已评估过的问题重复分析。

## 使用方式

将错误日志（完整 stderr + SampleInput 行）与以下类别逐一匹配。每类有**泛化匹配规则**（正则/关键字）和**排除条件**（避免误判）。

匹配任意一条 → 终止诊断，输出"已知不支持"结论。

---

## 类别 1：CANN dtype 不支持

**泛化规则**：

```
not implemented for DT_\w+
dtype=torch\.\w+.*not (currently )?supported
dtype=torch\.\w+.*not implemented
(aclnn\w+ failed, error code 161002).*(not implemented for) DT_\w+
dtype support list.*does not include
```

**说明**：CANN/aclnn API 存在但缺少特定 dtype 实现。可能是 CANN 版本未升级，或该 dtype 在硬件上本不支持。

**判定**：⛔ cann-not-implemented（CANN dtype 缺失）

---

## 类别 2：硬件维度限制

**泛化规则**：

```
cannot be larger than \d+ dimensions
dimension.*exceeds.*limit
(>)\s*\d+\s*dimensions.*not supported
(tensor|input).*has\s+\d+\s*dim.*max\s+\d+
```

**说明**：NPU 硬件对张量维度有上限（如 >8D 不支持）。

**判定**：⛔ cann-not-implemented（硬件限制）

---

## 类别 3：CUDA 独有功能

**泛化规则**：

```
(only|exclusively) supported on CUDA
is only (available|implemented) for CUDA
Jiterator is only supported on CUDA
requires CUDA (backend|device)
CUDA only
cuda_only
```

**说明**：某些 PyTorch 功能（如 Jiterator）仅 CUDA 后端实现，NPU 无对应能力。

**判定**：⛔ cann-not-implemented（CUDA 独有，NPU 无对标能力）

---

## 类别 4：内存格式/布局限制

**泛化规则**：

```
(only|must be|requires?)\s+(contiguous|preserve_format)\s+(memory format|format|tensor)
(contiguous|preserve_format).*(supported|required)
(format|layout).*not (currently )?supported
only.*strided.*supported
channels_last.*not.*supported
```

**说明**：NPU 对非 contiguous 或特定 memory format（如 channels_last）的支持有限。

**排除条件**：如果同时出现 `contiguous()` 或 `.contiguous()` 调用建议，则该错误可能是**可修复的**（补 contiguous 守卫），不自动归类为不支持。

**判定**：
- 无排除条件：⛔ cann-not-implemented（格式限制）
- 有排除条件：→ 进入步骤 3 常规分析（可能是行为不一致）

---

## 类别 5：特定未实现算子（白名单）

**泛化规则**：

```
aten::_scaled_dot_product_flash_attention
aten::_fill_mem_eff_dropout_mask_
aten::_fused_sdp_choice.*not (implemented|supported)
aten::_flash_attention_
aten::_efficient_attention_
aten::_scaled_dot_product_
(Jiterator|jiterator) is only supported on CUDA
```

**说明**：已评估过不可修复的特定 aten 算子。此类白名单应逐步积累。

**注意**：如果错误消息是 `"operator 'aten::xxx' is not currently supported on the NPU backend"` 且有 fallback 告警 → 不要匹配此类别，走步骤 3 判定流程（可能是可组合的）。此类别**仅**适用于已确认过"不可组合 + CANN 无对应 API"的 operator。

**判定**：⛔ cann-not-implemented（已评估，算子不可组合，CANN 无对应 API）

---

## 类别 6：test/ 用例自身的设备限制

**泛化规则**：

```
(only|exclusively).*(runs? on|supported on|tested on) (CUDA|CPU)
(onlyOnCUDA|onlyOnCPU)
device_type.*(not in|not found)
onlyAccelerator.*skip
```

**说明**：测试用例本身标注了设备限制。错误栈在 `test/` 或 `testing/_internal/` 中，未涉及 `torch_npu` 帧。

**判定**：⚠️ 测试框架限制（不可在 torch_npu 侧修复，需 skip 或用 XpuOnly 变体覆盖）

---

## 匹配流程

```
步骤 2 错误日志提取完成后：

1. 将错误消息 + SampleInput 行拼接为搜索文本
2. 按类别 1→6 顺序匹配
3. 第一个命中的类别 → 输出对应判定
4. 全部不命中 → 进入步骤 3 常规优先级路由
```

## 示例匹配

| 日志片段 | 匹配类别 | 判定 |
|---------|---------|------|
| `aclnnAbs failed, error code 161002 ... not implemented for DT_COMPLEX` | 类别 1 | dtype 不支持 |
| `RuntimeError: tensor has 9 dimensions, max is 8` | 类别 2 | 硬件限制 |
| `Jiterator is only supported on CUDA` | 类别 3 | CUDA 独有 |
| `RuntimeError: Only contiguous_format or preserve_format is supported` | 类别 4 | 格式限制 |
| `aten::_scaled_dot_product_flash_attention` | 类别 5 | 算子不可组合 |
| `TestSWAUtils has no attribute device_type` | 类别 6 | 测试限制 |
| `DispatchStub: missing kernel for npu` | 无匹配 | → 步骤 3 |
| `operator 'aten::linalg_householder_product' is not currently supported` | 无匹配 | → 步骤 3 |
