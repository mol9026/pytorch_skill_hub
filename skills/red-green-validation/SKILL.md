---
name: red-green-validation
description: >-
  修复隔离验证协议（红绿灯）：单撤一个修复、跑回归门禁、与全修复态基线对照，
  判定该修复是否真的必要。产出 verdict JSON 落盘证据。
  当一个 bug 的修复即将定案/提 PR 前，或多项修复同时生效需要逐一确认贡献时使用。
  Use before finalizing any fix or submitting a PR: isolate-revoke one fix,
  run the regression gate, compare against the all-fixes baseline, and
  record a verdict JSON. Mandatory when multiple fixes landed together.
---

# red-green-validation：修复隔离验证协议

## 概述

**为什么必须有这个流程**：两个教训换来的。rc2、rc3 两个"修复"都是在多根因同时
修复、全绿的状态下定案的——后来 verdict 实验才发现它们各自零贡献（rc3：transformers
5.17.0 已自动 cast fp32；rc2：aclnn FA v3 本就支持 strided），历史证据是 rc1 未修复
期的测量污染。**绿灯 ≠ 每个修复都有必要**。

## 触发条件（满足任一即必须跑）

1. 一个修复即将提 PR（提 PR 前的最后一道验证）
2. 多个修复同时生效，任何一项没被单独验证过
3. 历史证据（尤其是"修复前测到的错误"）来自其他修复未就位时期

## 协议（五步）

### 1. 建基线（绿灯态）

全修复在位，跑对应门禁，记录数值：

```bash
python3 scripts/precision_watchdog.py --gate <gate> --json $NPU_EVIDENCE_DIR/<fix>_green_baseline.json
# 或 doctor L1（模型级）：cos + bitwise
```

### 2. 隔离撤销（红灯态）

**只撤目标修复，其余全部保留**。撤销方式按修复形态：

| 修复形态 | 撤销方式 |
|---------|---------|
| fx_passes/codegen 源码守卫 | 注释掉守卫行（改动最小，便于恢复） |
| 注册类（monkeypatch/add_layout_constraint） | 调用点改 `pass  # disabled for red test` |
| 环境变量/配置 | 置反 |

**铁律**（2026-09-24 教训：漏一个触点 = 假阴性）：
- 按 PATCHES.md 枚举该修复的**全部触点**——双保险型修复（注册 + 调用点两层）必须两个都撤
- 改完必须 `rm -rf /tmp/torchinductor_root`（**缓存不感知 monkeypatch**，不清缓存读到旧产物 = 假绿灯）

### 3. 跑红灯

同样的门禁/用例，记录数值。合成用例 + 真实模型**各跑一组**（rc2 的教训：合成
strided 用例绿灯只证明方案有效，llama16h 全模型才证明必要性——反过来 K4 只跑
了直调就定案，被 rc1 污染）。

### 4. 对照判定

| 红灯 vs 基线 | 判定 | 动作 |
|-------------|------|------|
| 显著退化（cos 掉 / NaN / 非确定） | 修复**必要** | 恢复修复 → 提 PR（红绿两态数据进 PR body） |
| **完全一致** | 修复**不必要** | 归因修正：查历史证据为何误判（多半是当时别的根因未修）→ 取消 PR → 更新文档 |
| 部分退化 | 修复部分必要 | 细分场景再验证（如只反向退化） |

### 5. verdict 落盘（无论结论如何）

```json
{
  "experiment": "<fix> red-green isolation verdict",
  "date": "YYYY-MM-DD",
  "background": "修复的历史依据 + 为何现在隔离验证",
  "method": "撤销方式 + 保留的其他修复",
  "results": { "<实验名>": {"desc": "...", "cos": 1.0} },
  "env_check": {"torch_npu": "...", "cann": "...", "note": "排除环境升级解释"},
  "verdict": "必要 → 提 PR / 不必要 → 取消 + 归因修正",
  "followups": ["现场处置", "分支留存位置"]
}
```

落盘 `$NPU_EVIDENCE_DIR/<fix>_verdict.json`（evidence 只增不删——这是归因修正的
证据链）。

## 判定细则

- **环境核查先行**：红灯复现不出历史错误时，先核对 torch_npu/CANN 版本与时间戳
  （排除"库升级修好了"），再下"归因错误"结论
- **产物层取证**：红灯不红时查编译产物实锤（如 FA 输入是否有 `reinterpret_tensor`
  strided 直传、clone 有没有插）——"没插拷贝且结果对"比"结果对"更有说服力
- **历史证据没有留存脚本时**（如 rc2 的 K4 只在报告文字里）：坦白记录"当时构造
  无法完全还原"，以隔离实验的对照结果为准

## 完成后联动

- 修复不必要 → bug 报告加"归因修正"段（保留原机制链原文 + 修正结论，仿
  llama2_7b bug 报告 §9.3 处理）；PR_PLAN 标取消；PATCHES.md 触点标退役
- 修复必要 → 恢复现场，重跑门禁确认恢复，PR body 附红绿两态数据

## 实战案例

| 案例 | 日期 | 结论 | 证据 |
|------|------|------|------|
| rc3（inv_freq fp32 豁免） | 2026-09-29 | 不必要，取消 PR-5 | `pr5_verdict.json` |
| rc2（FA layout 约束） | 2026-10-09 | 不必要，取消（K4 的 0.158 系 rc1 期污染） | `rc2_verdict.json` + `rc2_red_llama16h.json` |
| rc4（reduction in-place） | 2026-09-24 | 必要（全撤 → NaN 复现） | watchdog 红灯记录，已提 PR #47261 |
