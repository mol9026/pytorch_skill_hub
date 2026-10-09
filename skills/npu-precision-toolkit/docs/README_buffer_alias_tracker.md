# buffer-alias-tracker（提案 D 交付）—— 产物内存复用审计

> 纯静态分析 `output_code.py`，自动检测 in-place reduction / reuse-after-write /
> 读改写链 等危险内存复用模式。把根因4（in-place reduction 跨核竞争）的
> 手工定位从约 1 小时压缩到秒级。

## 用法

```bash
# 单产物审计
python3 buffer_alias_tracker.py <output_code.py>

# 对比模式：绿色产物 vs 红色产物（定位修复前后差异）
python3 buffer_alias_tracker.py <red_artifact> --green <green_artifact>
```

输出：`buffer_alias_report.json` + stdout 摘要。

## 检测规则

| 规则 | 触发条件 | 风险等级 |
|------|----------|----------|
| `reduction+inplace` | reduction kernel 声明了 `mutated_arg_names` | high（根因4 核心） |
| `reduction-input-from-reuse` | reduction kernel 的输入来自 `# reuse` 复用 | high |
| `read-write-same-symbol` | 同一符号既作为 kernel 输入又被就地改写 | high |
| `raw-chain` | 符号作为前序 kernel 输入后，又被就地改写 | medium |
| `reuse-after-write` | `# reuse` 复用的是前面被写过的 buffer | medium |

## 验证结果（2026-09-24）

### 绿色产物（修复态）

```
alerts: 0
```

### 红色产物（rc1+2+4 全撤 + 清缓存）

```
[red] reduction+inplace: reduction kernel (1 reduction) 声明 mutated 参数 ['in_out_ptr0']，
对应调用参数 ['buf33'] —— 在 group dispatch 下存在并发竞争风险
```

`buf33 = buf0; del buf0  # reuse` 后紧跟最终 RMSNorm kernel 调用，
`buf33` 同时作为输入和 `in_out_ptr0` 被就地改写——这正是根因 4 的案发位置。

## 与提案 A/B/C 的协同

```
watchdog: 每日门禁 —— "还对不对"
doctor:   问题定位 —— "坏在哪一层/哪个 kernel"
kernel_isolate: 定点验证 —— "这个 kernel 重放是否稳定"
buffer_alias_tracker: 静态审计 —— "这个产物内存复用是否危险"
```

## 文件位置

| 文件 | 位置 |
|------|------|
| 主程序 | `/home/z00599499/scripts/buffer_alias_tracker.py` |
| 对比报告 | `/home/z00599499/evidence/llama2_7b/buffer_alias_compare.json` |

## 已知边界

1. 静态分析只看源码结构，**不跑真实模型**，所以检测是近似的（会漏但不会误报高危）。
2. 当前规则基于 llama2_7b 四根因提炼，新模型出现新模式时需追加规则。
3. 对比模式需要同时有"修复态"和"故障态"两个产物；日常 CI 可先用单产物模式。
