# 常见问题修复指导

> 本文件为诊断结论为"可修复"时提供修改建议参考。**只读诊断技能可参考本文件在报告中提供修复伪代码/代码示例，但不得直接修改本地机器上的代码文件。** 修复代码由 `torch-npu-issue-pipeline-core` 的实施阶段执行。

## 问题 1：算子回退到 CPU — 修复步骤

**解决步骤**：

1. 在 `npu_native_functions.yaml` 的 `supported` 部分添加算子声明：
```yaml
- func: xxx.out
  op_api: True
```

2. 在 `torch_npu/csrc/aten/ops/op_api/` 创建 kernel 文件 `XxxKernelNpu.cpp`

3. 实现算子逻辑，优先使用 CANN API：
```cpp
EXEC_NPU_CMD(aclnnXxx, ...);
```

4. 若无对应 CANN API，用现有算子组合实现

**示例**：Issue 251（conj_physical 算子）
- 修改 `npu_native_functions.yaml` 添加 `conj_physical.out` 和 `conj_physical_`
- 创建 `ConjPhysicalKernelNpu.cpp`，用 `aclnnComplex` 组合实现

## 问题 2：接口不存在 — 修复步骤

**解决步骤**：

1. **优先尝试替换接口**：将 `_cuda_xxx` 替换为 `_npu_xxx`
2. **若 NPU 接口不存在**，需在 `torch_npu/csrc/` 中添加：
   - `npu/Module.cpp` — 添加 Python 绑定函数
   - `core/npu/interface/AclInterface.h` — 添加接口声明
   - `core/npu/interface/AclInterface.cpp` — 添加接口实现
3. **若仅需修改测试用例**，添加 CUDA 环境模拟代码：
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch._C._cuda_setStream = torch_npu._C._npu_setStream
torch.cuda.get_device_capability = lambda *args, **kwargs: (10, 0)
torch._C._cuda_setDevice = torch_npu._C._npu_setDevice
torch._C._cuda_getCompiledVersion = lambda: 11080
torch.version.cuda = '11.8'
CUDA_VERSION = 11080
```

## 问题 3：类型检查失败 — 修复步骤

按「问题 1」处理，补充缺失的算子实现。

## 问题 4：编译失败 — 修复步骤

按编译指南操作，根据编译报错修改代码后重新编译。

## 问题 5：Ran 0 tests / 测试未执行 — 修复步骤

1. 在测试文件 `import torch` 后添加 CUDA 环境模拟代码（见问题 2 中的代码段）
2. 注释掉 `@unittest.skipIf` 装饰器（如果存在）
3. 注释掉 CUDA 相关导入：`# from jit.test_cuda import TestCUDA`

## 问题 6：测试被跳过 (CUDA not available) — 修复步骤

编辑测试文件，注释掉触发 CUDA 可用性检查的导入语句。

## 问题 7：CPU 侧不支持 Half 数据类型 — 修复步骤

在 NPU 侧实现该算子，避免回退 CPU。参考「问题 1」的步骤。
