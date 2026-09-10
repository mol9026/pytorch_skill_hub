# 策略：算子未适配（Missing Dispatch）— 诊断参考

> 本文件为只读诊断技能提供判定规则和搜索命令。修复指导内容见 `references/missing-dispatch-guide.md`。

## 根因识别特征

出现以下任意一种错误特征，判定为算子未适配：

- `NotImplementedError: Could not run 'aten::xxx' with arguments from the 'PrivateUse1' backend`
- `RuntimeError: no kernel found for 'aten::xxx' with device type 'npu'`
- `not implemented for 'PrivateUse1'`
- 告警：`operator 'aten::xxx' is not currently supported on the NPU backend and will fall back to run on the CPU`
- `RuntimeError: CAUTION: The operator 'aten::xxx' is not currently supported on the NPU backend.`（带或不带 `ERR01007 OPS feature not supported`）
- 测试通过但设备变成 CPU（`tensor.device` 为 `cpu` 而非 `npu`）

## 排除检查：fallback 是否为真正根因

当错误日志同时出现「fallback to CPU 告警」和以下任意特征时，fallback 可能只是 **side effect**，真正根因是底层基础设施问题（behavior-mismatch），而非算子缺失：

| 特征 | 含义 | 实际根因示例 |
|------|------|-------------|
| `aclrtMemcpy` 帧 | NPU 内存拷贝失败 | `copy_` 收到 null data_ptr |
| `CopyKernel` / `copy_` 帧 | copy 基础设施 | `copy_` 未处理 ZeroTensor |
| `Null_Pointer` / `data_ptr.*null` | 空指针传入 | ZeroTensor 的 `data_ptr=null` 被直接传给 aclrtMemcpy |

**检查流程**：

```
fallback to CPU 告警
    │
    ├── 错误栈中有 aclrtMemcpy / copy_ / Null_Pointer？
    │   ├── 是 → 先按 behavior-mismatch 检查：
    │   │        1. 定位 copy_ 调用的上游对应实现
    │   │        2. 逐行对比守卫分支（_is_zerotensor、numel==0 等）
    │   │        3. 若发现缺失分支 → 在 copy_ 基础设施修复，回 behavior-mismatch
    │   │        4. 若确认是算子缺失 → 继续本策略
    │   └── 否 → 继续本策略
    │
    └── 继续 missing-dispatch 流程
```

> 穿刺验证案例：`_efficientzerotensor` fallback CPU → `aclrtMemcpy null pointer` → 根因是 `copy_` 缺 `_is_zerotensor()` 守卫 → behavior-mismatch，非 missing-dispatch。

## CANN 未实现终止规则

当确认以下条件**同时满足**时，诊断阶段必须终止，标记为不可修复：

1. **CUDA 有原生实现**（即 NPU 禁止走 CPU fallback）
2. **缺失算子依赖 CANN/aclnn 底层能力**（不是纯路由/组合逻辑）
3. **CANN/aclnn 中没有对应的底层算子接口**（op-plugin 中无实现，aclnn 无对标 API）
4. **无法通过已有 NPU 算子组合实现**（不存在功能等价的 NPU 算子组合方案）

**终止时处理**：

1. 在 `根因分析.md` 中记录诊断过程和结论，根因类型标注为：**CANN 算子未实现（不可修复）**
2. 明确列出缺失的 CANN 算子名称和期望功能
3. 记录已搜索过的所有替代方案及其不可行的理由
4. 建议向 CANN 团队提交算子需求

**判定搜索命令**：

```bash
# 检查 op-plugin 中是否有对应算子的 C++ 实现
rg "op_name" $TORCH_NPU_REPO/third_party/op-plugin/op_plugin/ops/ -l

# 检查 aclnn 接口是否存在
rg "aclnn.*OpName\|aclnnOpName" $TORCH_NPU_REPO/third_party/op-plugin/ -l

# 检查 RegisterNPU.cpp 中的注册
rg "op_name" $TORCH_NPU_REPO/torch_npu/csrc/aten/RegisterNPU.cpp

# 检查 npu_native_functions.yaml 中的声明
rg "op_name" $TORCH_NPU_REPO/torch_npu/csrc/aten/npu_native_functions.yaml
```

## 诊断搜索命令

```bash
# 从日志中提取算子名
rg "aten::" <复现输出日志>

# 搜索 Python 层适配（monkey-patch 注册）
rg "op_name" $TORCH_NPU_REPO/torch_npu/utils/ $TORCH_NPU_REPO/torch_npu/npu/

# 搜索 C++ dispatch 注册
rg "REGISTER_NPU_IMPL\|TORCH_LIBRARY_IMPL.*PrivateUse1\|m\.impl.*op_name" \
    $TORCH_NPU_REPO/torch_npu/csrc/aten/ $TORCH_NPU_REPO/torch_npu/csrc/framework/

# 搜索 op-plugin 层
rg "op_name" $TORCH_NPU_REPO/third_party/op-plugin/op_plugin/ops/

# 搜索 CUDA 后端的算子注册
rg "op_name" $PYTORCH_REPO/aten/src/ATen/native/cuda/ $PYTORCH_REPO/aten/src/ATen/native/quantized/cuda/

# 查看 native_functions.yaml 中的 dispatch 配置
rg "op_name" $PYTORCH_REPO/aten/src/ATen/native/native_functions.yaml -A 10

# 查看 torch_npu 的 dispatch 配置
cat $TORCH_NPU_REPO/torch_npu/csrc/aten/npu_native_functions.yaml
```

## 适配层判断

| 情况 | 建议适配层 |
|------|-----------|
| 纯 Python 计算，无需 CANN 内核 | `torch_npu/utils/` Python 适配层 |
| 需要调用 CANN/ACL 算子（aclnn 接口） | `third_party/op-plugin/op_plugin/ops/` |
| 需要注册自定义 C++ dispatch | `torch_npu/csrc/aten/ops/` |
| 已有 op-plugin 实现但未在 torch_npu 注册 | `torch_npu/csrc/aten/npu_native_functions.yaml` |

## 代码路径参考

```
torch_npu 算子注册体系：
├── torch_npu/utils/                    # Python 层 monkey-patch 和适配
├── torch_npu/npu/__init__.py           # npu 命名空间初始化
├── torch_npu/csrc/aten/
│   ├── npu_native_functions.yaml       # dispatch 注册配置（YAML）
│   ├── VariableFallbackKernel.cpp      # fallback 处理逻辑
│   └── ops/                            # C++ dispatch 实现
└── third_party/op-plugin/op_plugin/
    ├── ops/opapi/                      # op-api 层算子（aclnn 接口）
    ├── ops/aclops/                     # 旧式 acl 算子
    └── utils/op_api_common.h          # EXEC_NPU_CMD 等核心宏
```
