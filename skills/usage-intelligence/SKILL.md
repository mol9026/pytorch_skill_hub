---
name: usage-intelligence
description: >-
  Claude Code 会话用量与成本优化技能：监控上下文占用、在合适时机提醒压缩、
  会话分型指导、识别可 skill 化的重复模式，并全程记录收益账本。
  当用户说"写状态"、询问上下文/压缩/token 成本、"查收益"、需要会话分型建议、
  或要求把重复操作封装成 skill 时使用。
  Use when "写状态"/"查收益" is said, or context usage, compaction, token cost,
  session strategy, or skill-harvesting is discussed.
---

# usage-intelligence：会话用量与成本优化（团队共享薄壳）

## 这是什么

本目录是**团队发现层**的薄壳入口。完整实现（hooks 引擎 + 行为协议）位于本仓库的
`.claude/skills/usage-intelligence/`，因为 hooks 必须挂在 `.claude/settings.json`
引用的本机路径上，无法直接搬到 `skills/`。

## 团队成员如何启用（安装到自己的项目）

1. 拷贝运行时组件到你的项目根：

   ```bash
   # 假设你的项目在 ~/myproj，本仓库在 ~/pytorch_skill_hub
   cp -r ~/pytorch_skill_hub/.claude/skills/usage-intelligence ~/myproj/.claude/skills/
   cp -r ~/pytorch_skill_hub/.claude/usage-intelligence   ~/myproj/.claude/
   ```

2. 把 hooks 注册进 `~/myproj/.claude/settings.json`（SessionStart / UserPromptSubmit /
   PreCompact 三个事件，完整配置见实现目录的 SKILL.md §hooks 注册段）。

3. 重启 Claude Code 会话，说「写状态」「查收益」验证生效。

## 核心命令（任何项目里装好后可用）

| 短语 | 动作 |
|------|------|
| 「写状态」/「保存状态」 | 把当前任务线索固化到 `.claude/usage-intelligence/session-state.md`，压缩/换会话零信息损失 |
| 「查收益」/「统计」 | 运行 `python .claude/skills/usage-intelligence/scripts/ui_core.py report` 并解读累计节省 |
| 「封装」 | 响应 skill 化提醒，把重复操作沉淀为正式 skill |

## 为什么值得装

实测单个长会话（1M 上下文压到 5-7%）每轮输入节省数十万 token；账本自动累计每次
压缩的实际收益。适合所有跑长任务的同事——尤其是 NPU 排查这类多天、多会话的工程。

## 边界

- hooks 引擎零依赖（纯 Python 标准库），不采集任何外发数据，账本只在本地
- 与 `skills/skill-harvest`（归集判定）配合：本 skill 负责"发现重复模式并提醒"，
  skill-harvest 负责"按标准模板收编进本仓库"
