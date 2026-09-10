# 示例：常规用例诊断（算子未适配）

## 输入

```
用例名：test_fn_gradgrad_scatter_reduce_mean_npu_float64
结果：FAIL
```

## 日志片段

```
RuntimeError: aclnnScatterReduce or aclnnScatterReduceGetWorkspaceSize not in libopapi.so, or libopapi.so not found.
ERR01004 OPS invalid pointer
SampleInput(input=Tensor[10,5], index=Tensor[5,5](int64), src=Tensor[5,5](float64), reduce='mean', include_self=False)
```

## 预期输出

- 根因类型：算子未适配（CANN API 不存在）
- 可修复性：✅ 可修复（模式 C2，scatter_add_ + 算术组合）
- 报告路径：`issues/test_fn_gradgrad_scatter_reduce_mean_npu_float64/根因分析.md`
