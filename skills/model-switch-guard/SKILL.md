---
name: model-switch-guard
description: >-
  模型切换守卫：从 CC Switch 网关日志观测真实的模型路由变化，区分
  主链切换（有缓存损失）、失败兜底重试、网关抖动、subagent 路由四类事件，
  对有损失的主链切换给出上下文规模与缓存风险提醒，并汇总月度报表。
  当用户询问"切换模型会损失多少缓存"/"模型怎么路由的"/"为什么 token 消耗异常"、
  或要审计模型使用与切换历史时使用。
  Use when auditing model routing/switches, estimating prompt-cache loss from
  model switches, or explaining abnormal token consumption via the CC Switch
  gateway logs.
---

# model-switch-guard：模型切换守卫

## 概述

**解决的问题**：多模型网关环境（CC Switch 等）下，模型切换会让 prompt cache
全部作废——1M 窗口上下文一次切换可损失数十万计费 token。但"切换"有四种性质
完全不同的形态，笼统报警天天误报。

**单一观测点原则**：不监听 `/model` 命令、不监听网关 UI——原因有无数种
（手动切换、供应商变更、配置修改、兜底重试），**出口只有一个**：实际请求的路由
结果。CC Switch 的 `proxy_request_logs` 表记录每条请求的 `request_model`（客户端
要的）→ `model`（实际路由的）→ 状态码 → 上下文规模，是唯一可靠信号源。

## 数据源与前提

- `~/.cc-switch/cc-switch.db`（SQLite，只读打开；env `CC_SWITCH_DB` 可覆盖）
- 网关需启用 local proxy（CC Switch `enableLocalProxy: true`）
- 四类事件的区分依据全部来自本机 23 天 4256 请求的实测（见 §实测依据）

## 四类事件语义（为什么不是"所有切换都报警"）

| 事件 | 判定 | 处置 | 实测依据 |
|------|------|------|---------|
| **PROVIDER_SWITCH** | 主链模型变化（排除白名单与失败请求） | ctx≥50k 时提醒缓存损失；记账 | 真·损失源：23 天 4 次，全部可复盘到人工操作 |
| **RETRY_FALLBACK** | 失败（非 200）后 ≤120s 内换目标重发 | 记账，不报（用户已在等） | 12 次，403/429 兜底重试 |
| **GATEWAY_JITTER** | 同 request_model 秒级来回 | 静默 | 网关对辅助请求的内部分段路由 |
| **SUBAGENT** | 白名单模型（独立上下文/辅助实例） | 静默 | 见下"白名单的实测依据" |

### 白名单的实测依据（23 天数据，误报过滤器）

- `MiniMax-M3`：subagent 配置路由（`CLAUDE_CODE_SUBAGENT_MODEL`），每请求 <1.2k token，独立上下文——**不伤主会话缓存**
- `glm-5.3`（无后缀短名，区别于主力 `glm-5-3-260814`）：网关把部分预检/分类请求路由的辅助实例——其出现点**时间上嵌入**主链流（前后 2-70s 内 glm-5-3 请求照常），是分段不是替代。52 次出现若不滤掉会产生 59 次假切换
- `qwen3.8-flash`：haiku 档映射的辅助调用

### 失败请求不进主链（另一个实测坑）

网关映射失败（上游 429/403/502）时，日志的 `model` 字段会**回显 request_model
原名**（如 `claude-sonnet-4-6` + status 429 + ctx=0）——不是真实路由。不滤掉会把
每次限流当成一次"切回官方模型"。

## 使用

```bash
# 全量事件流（JSON）
python skills/model-switch-guard/scripts/detector.py scan

# 增量（自某 unix 时间戳）
python skills/model-switch-guard/scripts/detector.py scan 1791500000

# 汇总报表（近 N 天；无参=全部）
python skills/model-switch-guard/scripts/detector.py report 30
```

报表输出：各事件计数、高上下文切换次数、`switch_ctx_tokens_at_risk`
（风险上下文总量——实际计费损失 ≈ 0.9×该值，因缓存读价约为全价 10%）。

## 回归验证（安装后跑一次）

对 2026-09-20 ~ 10-12 的历史数据，期望输出（若期间无新操作）：
`PROVIDER_SWITCH=4`（kimi→glm ×2、glm→kimi ×1、glm→flash ×1，均为人工/CC Switch
操作），`RETRY_FALLBACK≈12`，无假 PROVIDER_SWITCH。

## Hook 挂接（可选，自动化提醒）

在 SessionStart hook 里跑 `detector.py report 1`（近 1 天），若
`switch_alert_count > 0` 则把提醒注入会话上下文；每次检测后记录已处理的最大
时间戳（存 `.claude/usage-intelligence/` 下），实现增量。与 usage-intelligence
的账本打通：PROVIDER_SWITCH 事件附带的 ctx 数计入其收益账本的"避免损失"侧。

## 局限

- 依赖 CC Switch 日志格式（表名/字段变更需跟版）；其他网关（自建 LiteLLM 等）
  需适配数据源适配层
- 风险估算用固定 0.9 系数（缓存读 1 折价假设）；若供应商定价不同，改
  `report()` 注释里的系数
- `ALERT_CTX_THRESHOLD=50k` 以下不提醒：小上下文切换损失 <4.5 万等值 token，
  提醒噪声大于收益（可 env 调）
