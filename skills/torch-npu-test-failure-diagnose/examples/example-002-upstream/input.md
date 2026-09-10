# 示例：上游/测试框架问题

## 输入

```
用例名：test_averaged_model_mixed_device
结果：FAIL
```

## 日志片段

```
AttributeError: 'TestSWAUtils' object has no attribute 'device_type'
common_device_type.py:2000  only_fn → self.device_type in ("cpu", "meta")
```

## 预期输出

- 根因类型：上游问题（测试框架适配）
- 可修复性：⚠️ 不可在 torch_npu 侧修复
- 报告路径：`issues/test_averaged_model_mixed_device/根因分析.md`
