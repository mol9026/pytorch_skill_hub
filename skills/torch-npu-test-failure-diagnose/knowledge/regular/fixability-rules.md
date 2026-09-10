# CANN 可行性判定规则

> 用于"算子未适配"根因类型的可修复性判定。

## 判定前提

只适用于**算子未适配**根因。如果日志中的错误不是算子缺失（是行为不一致/内存/环境问题），直接判定为"可修复"。

## 判定流程（三级）

### 第一级：CUDA 对照

```
该算子在 CUDA 上是否有原生实现？
├── 无（CUDA 也走 fallback 或同样失败）
│   → ✅ 可修复：NPU 走 CPU fallback 是对齐行为
│   → 方向：确保 fallback 回传路径完整即可
│
└── 有（CUDA 正常执行）→ 进入第二级
```

### 第二级：缺失类型分析

从日志中提取错误的精确性质：

| 日志特征 | 缺失类型 | 含义 |
|---------|---------|------|
| `not in libopapi.so` / `ERR01004 OPS invalid pointer` | **CANN API 不存在** | CANN 版本根本没这个接口 |
| `aclnnXxx failed, error code 161002` + `not implemented for DT_DOUBLE` / `dtype support list [...]` | **dtype 不支持** | CANN API 存在，但此 dtype 不支持 |
| `not currently supported on the NPU backend` / `no kernel found` / `will fall back to run on the CPU`（无 aclnn 错误） | **dispatch 缺失** | 上层路由未注册，底层能力未知 → 需进一步搜索 |

### 第三级：组合可行性判定

```
该算子能否用已有 NPU 算子组合实现？
├── 可组合
│   → ✅ 可修复（模式 C2）
│   → 方向：Python torch.library.impl 注册，用 scatter_add_/算术/
│            索引/matmul 等已有原语组合
│   → 示例：
│       scatter_reduce(mean) = scatter_add_ + count + 除法
│       dot(a,b) = (a * b).sum()
│       masked_fill = where(mask, value, self)  # 如果 where 支持
│
└── 不可组合
    → ⛔ cann-not-implemented（不可修复，终止）
    → 原因：该算子是底层原语，无法用加减乘除分解
    → 示例：
        sort：排序是原子操作
        matmul：矩阵乘法是 Cube 核心
        fft：傅里叶变换依赖专用算法
        convolution：依赖 im2col+gemm 专用路径
```

## 典型场景速查

| 场景 | 判定 | 理由 |
|------|------|------|
| aclnnSort 不支持 float64 | ⛔ 不可修复 | sort 不可组合 |
| aclnnAbs 不支持 complex128 | ✅ 可修复 | abs(c)=sqrt(real²+imag²)，可组合 |
| aclnnAngleV2 不支持 complex128 | ✅ 可修复 | angle = atan2(imag, real)，可组合 |
| aclnnScatterReduce 不存在于 libopapi | ✅ 可修复 | scatter_reduce 可用 scatter_add_ 组合 |
| aclnnDot 不支持 float64 | ✅ 可修复 | dot(a,b) = (a*b).sum() |
| aclnnMatmul 不支持 float64 | ⛔ 不可修复 | matmul 是 Cube 底层原语 |
| aten::linalg_householder_product 走 CPU fallback | ✅ 可修复 | 可在 Python 层实现分解 |
| aten::_convolution 不支持 dilated | ⛔ 不可修复 | 卷积是底层原语 |

## 判定后的输出

### ✅ 可修复
```
根因分析.md 中标注：
- 可修复性：✅ 可修复
- 修复方向：{模式 C2 组合 / 模式 B YAML 注册 / 模式 C op-plugin 新增}
- 后续建议：进入 diagnose 实施阶段
```

### ⛔ cann-not-implemented
```
根因分析.md 中标注：
- 可修复性：⛔ 不可修复（cann-not-implemented）
- 缺失的 CANN 能力：{aclnnXxx 的 YY dtype 支持 / aclnnYyy API}
- 终止原因：{具体说明哪条不满足}
- 后续建议：向 CANN 团队提交算子需求
- 不创建修复代码、不进入 verify
```
