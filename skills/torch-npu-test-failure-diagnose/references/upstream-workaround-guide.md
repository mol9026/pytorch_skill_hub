# 上游问题 Workaround 指导

> 本文件为诊断结论为"上游问题"时提供 Workaround 建议参考。**只读诊断技能可参考本文件在报告中提供修复伪代码/代码示例，但不得直接修改本地机器上的代码文件。** 修复代码由 `torch-npu-issue-pipeline-core` 的实施阶段执行。

## Workaround A：在 torch_npu 测试中添加 skip 标记

**适用场景**：测试直接在 `torch_npu/test/` 中有对应用例。

```python
import unittest
import torch_npu

class TestSomething(unittest.TestCase):

    @unittest.skip(
        "SKIP: upstream PyTorch bug in torch.some_op, "
        "tracking: https://github.com/pytorch/pytorch/issues/XXXXX. "
        "TODO: remove this skip after upstream fix is merged and torch_npu rebases."
    )
    def test_some_op_npu(self):
        # ...
        pass
```

## Workaround B：条件 skip（只在特定版本跳过）

```python
import torch
import unittest

# 在特定 torch 版本区间跳过（已知该版本有 bug）
@unittest.skipIf(
    torch.__version__ < "2.8.0",
    "Upstream bug fixed in torch 2.8.0, "
    "see https://github.com/pytorch/pytorch/issues/XXXXX"
)
def test_some_op_npu(self):
    pass
```

## Workaround C：在 torch_npu 适配层添加保护

**适用场景**：上游函数的某个分支有 bug，可以在 NPU 适配层绕过。

```python
# torch_npu/utils/some_utils.py

def _npu_safe_some_op(input, *args, **kwargs):
    """
    NPU 侧的临时适配，绕过 upstream torch.some_op 的已知 bug。
    上游 issue：https://github.com/pytorch/pytorch/issues/XXXXX
    TODO：在 upstream 修复并 rebase 后移除此适配。
    """
    # 临时替代实现
    return some_alternative_impl(input, *args, **kwargs)
```

## Workaround D：patch 上游函数（最后手段）

```python
# torch_npu/__init__.py 或 torch_npu/npu/__init__.py

import torch
from torch_npu.utils._upstream_patches import _patched_some_op

# 只在 NPU 可用时 patch，不影响 CUDA/CPU 路径
if torch.npu.is_available():
    torch.some_op = _patched_some_op
    # 上游 issue：https://github.com/pytorch/pytorch/issues/XXXXX
```

## 上报流程

### 上报前的准备

```markdown
# 上游问题报告草稿

## 问题描述
简述：在什么场景下，预期行为是什么，实际行为是什么。

## 复现脚本
```python
import torch

# 最小复现代码（不依赖 torch_npu）
x = torch.randn(4, 4)
result = torch.some_op(x)
# 预期：...
# 实际：RuntimeError: ...
```

## 环境信息
- PyTorch 版本：X.Y.Z
- Python 版本：3.X
- 操作系统：Linux
- 是否在 CUDA 上复现：是/否

## 附加信息
- 相关代码文件：torch/xxx.py:Line N
- 调用栈：...
```

### 上游 PR/Issue 上报渠道

1. **GitHub Issues**（推荐）：https://github.com/pytorch/pytorch/issues
   - 标签：`bug`、`module: xxx`（对应模块）
   - 优先搜索是否已有相同问题

2. **GitHub PR**（如果已有修复方案）：https://github.com/pytorch/pytorch/pulls

3. **内部问题跟踪**（如有）：在公司内部系统中关联上游 issue

## 与社区协作的注意事项

1. **先搜索再上报**：在上报前，搜索 GitHub Issues 是否已有相同问题，避免重复提交。

2. **提供最小复现**：上游社区优先响应有最小复现脚本的 issue，不要把 torch_npu 专用代码放进去。

3. **不要修改上游 PyTorch 源码**：即使在本地修改了 `pytorch/` 目录下的代码进行调试，不要把这些修改提交到 torch_npu 仓库。

4. **Workaround 要有明确的撤销条件**：添加的临时 skip 或 patch 必须注明对应的上游 issue/PR 链接和移除条件。

5. **NPU 侧和上游侧要同步跟踪**：在 torch_npu 的 issue 跟踪系统中关联上游 issue，确保上游修复后能及时跟进。

6. **区分"NPU 特有"和"上游 bug"**：有些问题看似上游 bug，但实际上是 torch_npu 的适配没有考虑边界情况，要仔细判断。
