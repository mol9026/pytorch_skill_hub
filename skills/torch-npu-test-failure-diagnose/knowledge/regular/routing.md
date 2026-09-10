# 根因路由规则

> 用于诊断阶段：根据日志中的错误特征，确定根因类型。

## 路由表

| 错误特征 | 根因分类 |
|---------|---------|
| `NotImplementedError` / `no kernel found` / `aten::xxx not implemented` / `fallback to CPU` 告警 / `not supported on the NPU backend` / `aclnnXxx failed, error code 161002` / `not in libopapi.so` (ERR01004) | **算子未适配** |
| 精度误差超阈值（`AssertionError: not close`）/ shape 不匹配 / dtype 不支持 / 结果值不一致 / GradcheckError | **算子行为不一致** |
| OOM / `ENOMEM` / `out of memory` / `EI0004` / 内存峰值异常大于 CUDA 基线 | **内存问题** |
| 编译失败 / `ImportError` / `ModuleNotFoundError` / 环境缺依赖 / 版本不匹配 / `libxxx.so: cannot open` | **环境/编译问题** |
| CUDA 上同样失败 / `torch` 原生代码 bug / 错误栈无 torch_npu 帧 / 需要上游修复 | **上游问题** |

## 多特征冲突优先级

1. **环境/编译问题优先**：`import torch_npu` 失败 → 先解决环境
2. **上游问题次之**：CUDA 同样失败 → 非 NPU 适配问题
3. **算子未适配 > 算子行为不一致**：fallback 告警 + 精度误差 → 先补算子
4. **内存问题独立处理**：OOM 需单独内存分析流程
5. **级联缺失识别**：>= 2 个不同 `aten::xxx not supported` → 级联缺失问题

## fallback 告警注意

`fallback to CPU` 可能是 side effect 而非根因。若错误栈同时出现 `aclrtMemcpy` / `copy_` / `Null_Pointer` 帧，应先检查 copy 基础设施对齐（行为不一致），再判定为算子缺失。

## CANN 错误码速查

| 错误码 | 含义 |
|--------|------|
| EZ1001 | AclNN 参数错误（通常 = dtype/format 不支持） |
| ERR00100 | PTA call acl api failed（算子调用失败） |
| ERR01004 | OPS invalid pointer（API 不存在于 libopapi.so） |
| EI0004 | 内存不足 |
