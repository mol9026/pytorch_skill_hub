# 策略：上游 PyTorch 问题（Upstream Issue）— 诊断参考

> 本文件为只读诊断技能提供判定规则。Workaround 指导内容见 `references/upstream-workaround-guide.md`。

## 根因识别特征

满足以下条件时，判定为上游 PyTorch 问题：

- **核心判定依据**：将测试中的 `device='npu'` 替换为 `device='cuda'`（或 `device='cpu'`），在相同条件下同样失败
- PyTorch 原生代码（`torch/` 或 `aten/`）存在 bug，与 NPU 适配无关
- 错误发生在 PyTorch 的 Python 层（调用栈中没有 `torch_npu` 或 `op-plugin` 帧）
- 已知的 upstream issue，社区已有讨论或 PR

## 如何判定为上游问题

### 步骤 1：在 CUDA 上复现同一失败

这是最关键的判定步骤：

```bash
# 方法一：替换设备参数后运行
python -m pytest <测试文件> -k <测试名> --device=cuda

# 方法二：用 CPU 验证（更便捷，不依赖 CUDA 设备）
python -c "
import torch
x = torch.randn(4, 4, device='cpu')
result = torch.some_op(x)  # 如果这里也失败 -> 上游问题
"
```

### 步骤 2：分析调用栈，确认错误位置

上游问题的调用栈特征：

```
Traceback (most recent call last):
  File "test_xxx.py", line N, in test_xxx_npu
    result = torch.some_function(x, ...)
  File "/path/to/torch/functional.py", line N, in some_function
    return _VF.some_function(input, ...)    # ← torch 原生 Python 层
RuntimeError: Expected xxx but got yyy                # ← 错误在 torch 侧

# 注意：调用栈中没有 torch_npu 的帧
```

### 步骤 3：搜索 PyTorch 仓库是否已有相关 issue 或 PR

```bash
# 在本地 pytorch 源码中搜索相关注释或 TODO
rg "FIXME|TODO|BUG|HACK" pytorch/torch/ \
    -g "*.py" --context 2 | grep -A2 -B2 "some_function"

# 查看该函数近期变更
cd pytorch && git log --oneline -10 -- torch/functional.py
```

## 判定上游问题的反例（不应判定为上游）

以下情况看起来像上游问题，但实际上是 NPU 适配问题：

| 现象 | 实际原因 | 正确处理 |
|------|---------|---------|
| CPU 上 PASS，NPU 上 FAIL | NPU 适配缺失或错误 | 查 missing-dispatch 或 behavior-mismatch |
| CUDA PASS，NPU FAIL，错误在 torch_npu 调用栈中 | torch_npu 适配 bug | 查 behavior-mismatch |
| 精度差异（NPU vs CUDA） | NPU 硬件精度特性 | 查 behavior-mismatch |
| NPU OOM，CUDA 正常 | NPU 内存额外开销 | 查 memory-issue |

## 注意事项

1. **CUDA 环境不可用时的替代方案**：如果没有 CUDA 设备，用 CPU 做基准（大多数算子 CPU 和 CUDA 行为一致）。若 CPU 也失败，确认是上游问题。

2. **版本差异可能导致误判**：CUDA 上用的 torch 版本与 NPU 上不同时，要用同一版本对比。

3. **上游已修复但 torch_npu 未 rebase**：上游 PyTorch 已经修复了某个 bug，但 torch_npu 还在基于旧版本，此时需要等待 torch_npu rebase，或临时 cherry-pick 修复。
