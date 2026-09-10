# 常见问题分类与快速参考

> 本文件为只读诊断技能提供快速匹配。修复指导内容见 `references/common-issues-guide.md`。

## 问题 1：算子回退到 CPU

**现象**：`Warning: The operator 'aten::xxx' is not currently supported on the NPU backend and will fall back to run on the CPU.`

**根因**：NPU 后端缺少该算子的 kernel 实现，回退到 CPU 执行。跨设备传递时可能导致 autograd 或类型检查失败。

**诊断指向**：完整诊断流程见 `knowledge/regular/missing-dispatch.md`。

## 问题 2：接口不存在

**现象**：`AttributeError: module 'torch._C' has no attribute '_cuda_xxx'`

**根因**：测试用例使用了 CUDA 特定接口，NPU 环境中无对应接口。

**诊断指向**：优先级 2（上游问题）或优先级 3（算子未适配），取决于接口是否可在 NPU 侧实现。

## 问题 3：类型检查失败

**现象**：`isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED`

**根因**：通常由算子缺失引起。算子回退 CPU 后，跨设备传递时 autograd 类型检查失败。

**诊断指向**：按「问题 1」处理，补充缺失的算子实现。

## 问题 4：编译失败

**现象**：编译 torch_npu 时报错

**根因**：环境/编译问题

**诊断指向**：优先级 1（环境/编译问题），检查编译配置和依赖。

## 问题 5：Ran 0 tests / 测试未执行

**现象**：运行测试时 `Ran 0 tests in 0.000s`

**根因**：测试框架在 NPU 环境中未正确加载测试用例，通常是 CUDA 环境检查未通过。

**诊断指向**：优先级 2（上游问题），测试框架适配问题。

## 问题 6：测试被跳过 (CUDA not available)

**现象**：`CUDA not available, skipping tests`

**根因**：测试框架的 CUDA 可用性检查阻止了 NPU 上的测试执行。

**诊断指向**：优先级 2（上游问题），测试框架适配问题。

## 问题 7：CPU 侧不支持 Half 数据类型

**现象**：`"xxx_cpu_kernel" not implemented for 'Half'`，同时伴有算子回退 CPU 的警告。

**根因**：NPU 缺少算子实现 → 回退 CPU → CPU kernel 不支持 Half。

**诊断指向**：按「问题 1」处理，在 NPU 侧实现该算子，避免回退 CPU。
