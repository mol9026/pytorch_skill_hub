---
name: usage-intelligence
description: >-
  Claude Code 会话用量与成本优化技能：监控上下文占用、在合适时机提醒压缩、
  会话分型指导、识别可 skill 化的重复模式，并全程记录收益账本。
  当用户说"写状态"、询问上下文/压缩/token 成本、"查收益"、需要会话分型建议、
  或要求把重复操作封装成 skill 时使用。
  Use when the user says "写状态"/"查收益", asks about context usage, compaction,
  token cost, session strategy, or wants to turn repeated patterns into skills.
---

# usage-intelligence：会话用量与成本优化

## 概述

由 hooks 驱动的自动化层 + 本 skill 的行为协议组成。hooks 负责**检测与记账**（零 token、
不打扰），本 skill 负责**执行**（写状态文件、压缩前固化线索、skill 化流程）。

## 组件与文件位置

| 组件 | 位置 |
|------|------|
| Hook 引擎（Python，无依赖） | `.claude/skills/usage-intelligence/scripts/ui_core.py` |
| Hooks 注册 | `.claude/settings.json` |
| 用户规则（可编辑） | `.claude/usage-intelligence/rules.md` |
| 状态文件（"写状态"落点） | `.claude/usage-intelligence/session-state.md` |
| 收益账本 | `.claude/usage-intelligence/benefit-ledger.json` + `benefits.md` |

## 用户命令（识别这些短语）

| 短语 | 动作 |
|------|------|
| **「写状态」** / 「保存状态」 | 执行 §写状态协议 |
| **「查收益」** / 「统计」 | 运行 `python .claude/skills/usage-intelligence/scripts/ui_core.py report < /dev/null` 并解读 |
| **「封装」**（响应 skillify 提醒） | 执行 §skill 化流程 |
| 「上下文怎么样」/「占用多少」 | 快速回答占用率与趋势 |
| 「重置账本」 | 清空 benefit-ledger.json（确认后） |

## 写状态协议（核心保效果机制）

**触发**：用户说「写状态」/「保存状态」，或 PreCompact 前无新鲜状态文件时。

向 `.claude/usage-intelligence/session-state.md` 写入（覆盖）：

```markdown
# 会话状态快照（{日期时间}）

## 当前任务
- {一句话：在做什么、到哪一步}

## 活动线索（正在追、未落盘到报告的）
- {buffer/kernel 名、行号、假设清单、下一步实验设计}

## 关键文件（本轮会话动过的）
- {路径 → 作用}

## 恢复指引
- {下一个会话/压缩后：先读什么、从哪继续}
```

**纪律**：
- 正在追的线索**必须**先写状态再压缩（红区线 70% 时 hook 会拦截提醒）
- 状态文件 ≤50 行，超了说明该拆分了
- 写完回复用户一行：状态已写入 X，可以安全压缩

## 契约（不可违反）

1. **效果优先**：不得为省 token 而跳过必要的验证步骤；不得因压缩丢失正在追的线索
2. hook 注入的提示必须被简短确认（一句话），不展开；若与用户当前请求冲突，用户请求优先
3. hook 失败静默（exit 0）——工具绝不能阻塞用户输入
4. 状态文件是压缩/恢复的唯一可信锚点，不凭摘要记忆猜线索
5. 收益数字为估算口径（压缩节省 × 10 轮假设），向用户汇报时注明"估算"
6. 压缩后恢复工作：先读 rules.md 纪律 + session-state.md 线索，再继续

## skill 化流程（响应「封装」）

1. 从 `.claude/usage-intelligence/skillify_candidates.json` 提取相似任务的历史 prompt
2. 与用户确认封装目标（命令行工具 vs Claude skill）两选一
3. 若为 Claude skill：按本仓库 `skills/` 目录约定（YAML frontmatter：name/description，见 `skills/pr-review/SKILL.md`）创建 `skills/{name}/SKILL.md`
4. 新 skill 描述里写清触发短语（模仿现有 skill 的 description 写法）
5. 完成后在账本记一条 `kind: skillify-delivered`

## 分型表（hook 注入提醒时，按此执行）

| 分型 | 特征 | 上下文策略 |
|------|------|-----------|
| 排查/探索型 | 多轮实验、读大量产物/日志 | 允许高占用；Grep 定位后只读需要的行段；每阶段结束说「写状态」 |
| 修复/实施型 | 目标明确，改代码、打补丁 | 优先新开会话；只带补丁文件+报告相关章节；不重读全量报告 |
| 文档/汇报型 | 写报告、PR、案例分享 | 新开会话；只读报告相关段落；产物先落盘再润色 |
| 简单问答型 | 一问一答 | 无需策略 |

## 压缩决策表

| 条件 | 动作 |
|------|------|
| ≥50% 且自然断点（状态文件新鲜） | 建议 /compact，报估算节省 |
| ≥50% 但实验中途 | 先「写状态」再决定 |
| ≥70% 红区 | 必须先「写状态」再 /compact |
| <50% | 正常工作，无需动作 |

## 收益口径说明

- **compact-done**：压缩时点快照记账，`est_saved_total = 每轮节省 × 10 轮假设`
- **advice-shown**：提醒已展示（不直接计收益，采纳后由 compact-done 记账）
- 汇报时说明"估算口径"，避免误导。真实节省可用前后会话 token 总量差分核对

## 安装/移植（分享给团队）

1. 拷贝 `.claude/skills/usage-intelligence/`（Python ≥3.10，无第三方依赖）
2. 拷贝 `.claude/settings.json` 里的 `hooks` 段落到目标项目同名文件
3. 首次运行自动生成 `.claude/usage-intelligence/rules.md` 等状态文件
4. 团队成员无需任何配置，hooks 自动生效

## 排障

- hook 无输出：`python .claude/skills/usage-intelligence/scripts/ui_core.py monitor < test-input.json` 手测
- 注入内容乱码：确认 Python 为 3.7+（stdin/stdout reconfigure）
- 权限提示：settings.json 的 hooks 需要用户在 Claude Code 里确认一次
