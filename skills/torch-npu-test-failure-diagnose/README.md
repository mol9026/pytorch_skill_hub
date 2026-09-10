# torch-npu-test-failure-diagnose

NPU 用例失败**只读诊断**技能。根据用例名、执行结果和日志，进行根因分析并输出结构化诊断报告。

## 定位

是 `torch-npu-issue-pipeline-core` 流水线中 **diagnose 阶段的只读子集**：
- ✅ 覆盖：错误特征提取、根因分类、CANN 可行性判定、级联缺失检测
- ❌ 不覆盖：代码修复、编译、验证、交付

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| CANN | 8.0.RCx | 华为昇腾计算架构 |
| PyTorch | 2.1+ | 上游 PyTorch |
| torch_npu | 对应 CANN 版本 | NPU 适配层 |
| Python | 3.8+ | 运行环境 |
| Linux | x86_64 / ARM64 | NPU 驱动仅支持 Linux |

## 使用方式

在对话中提供以下信息即可触发分析：

```
用例名：test_fn_grad_sort_npu_float64
结果：FAIL
日志：...（粘贴执行日志）
```

或作为 slash 命令：`/torch-npu-test-failure-diagnose`

## 输入

| 输入 | 必须 | 说明 |
|------|------|------|
| 用例名 | ✅ | 完整参数化名称 |
| 执行结果 | ✅ | PASS / FAIL / SKIP / ENV_BLOCKED |
| 日志 | FAIL 时 | stderr/stdout 完整输出 |
| 环境 | ✅（自动获取） | 执行 `scripts/collect-env.sh` 采集，无需用户提供 |

## 输出

`issues/{case_name}/根因分析.md`，按 `templates/diagnosis-report.md` 模板填充，包含：
- 错误特征提取表
- 根因分类与路由依据
- CANN 可行性判定（可修复 / cann-not-implemented）
- 级联缺失检测（如适用）

## 知识文件

### 常规诊断 (`knowledge/regular/`)

| 文件 | 内容 |
|------|------|
| `routing.md` | 根因路由规则 + CANN 错误码速查 |
| `fixability-rules.md` | CANN 可行性三级判定流程 |
| `missing-dispatch.md` | 算子未适配诊断参考（排除检查、CANN 终止规则） |
| `upstream-issue.md` | 上游问题判定参考 |
| `diagnosis-pitfalls.md` | 6 类常见误判陷阱 |
| `common-issues.md` | 7 类常见问题分类与快速参考 |

### 通用 (`knowledge/common/` — 常规和分布式共同引用)

| 文件 | 内容 |
|------|------|
| `unsupported-patterns.md` | 6 类已知不支持模式泛化规则，含正则匹配和排除条件 |

### 分布式诊断 (`knowledge/distributed/`)

| 文件 | 内容 |
|------|------|
| `distributed-routing.md` | 分布式检测规则、扩展路由表（HCCL/Device Error）、HCCL 能力矩阵 |
| `distributed-pitfalls.md` | 6 类分布式特有误判陷阱 |

### 修复实施参考 (`references/`)

| 文件 | 内容 | 何时使用 |
|------|------|---------|
| `missing-dispatch-guide.md` | 算子未适配修复指导、反面案例、级联处理 | 诊断结论为"可修复"后，提供修改建议时 |
| `common-issues-guide.md` | 7 类常见问题修复指导 | 同上 |
| `upstream-workaround-guide.md` | 上游问题 Workaround 指导和上报流程 | 同上 |
