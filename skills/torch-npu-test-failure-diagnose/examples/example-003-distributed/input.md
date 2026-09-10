# 示例：分布式用例诊断

## 输入

```
用例名：test_sharded_embedding_rowwise
结果：FAIL
```

## 日志片段

```
RuntimeError: The operator 'aten::xxx' is not currently supported on the NPU backend
HCCS LINK ERROR on rank 1
watchdog timeout on rank 0
```

## 预期输出

- 分布式检测：是（Tier 1：用例名含 `sharded`）
- 优先级 1.5：HCCL 通信错误（根因）
- 优先级 3：算子未适配（side effect，通信恢复后可能消失）
- 报告路径：`issues/test_sharded_embedding_rowwise/根因分析.md`
