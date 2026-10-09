---
name: skill-harvest
description: >-
  工作成果自动归集技能：判断一项工作中沉淀的脚本/模板/方法论是否达到
  "应收编进 pytorch_skill_hub"的标准，并按标准结构生成 skill 草稿。
  当用户说"这个可以归集吗"/"收进 hub"/"做成 skill"，或 usage-intelligence
  发出 skill 化提醒且成果属于 NPU 工程类（而非 Claude 会话类）时使用。
  Use when deciding whether accumulated scripts/templates/methodology from
  NPU work should be harvested into the team skill hub, and to scaffold
  the skill directory following house conventions.
---

# skill-harvest：成果归集与 skill 脚手架

## 概述

解决的问题：工作过程中自然沉淀的脚本、模板、方法论散落在服务器目录和会话上下文里，
没有统一入口就没人复用。本 skill 定义"什么值得收、按什么结构收"，并半自动生成
符合本仓库惯例的 skill 目录草稿。

与 `usage-intelligence` 的分工：那个负责**发现重复模式并提醒**（会话层），本 skill
负责**按团队标准收编**（仓库层）。提醒来了 → 用户说「归集」→ 走本流程。

## 归集判定（先过三关，再动手）

### 第一关：价值判定（至少满足一条）

| 信号 | 说明 |
|------|------|
| **重复 ≥ 3 次** | 同一操作/判断在 ≥3 个不同对象上执行过（如 13 个穿刺模板） |
| **明确契约** | 有可复述的输入输出约定（如 `build_args(device) -> (args, ref, model, forward_fn)`） |
| **踩坑沉淀** | 含"不这么做会浪费 X 时间"的教训（如缓存不感知 monkeypatch） |
| **跨人复用** | 团队成员问过/会问的问题（如"这个模型图模式能跑吗"） |

### 第二关：稳定性判定（全部满足）

- 流程已收敛——最近一次执行没有再改步骤（还在演进的流程先记台账不归集）
- 有验证手段——能一句话说出"怎么证明这个 skill 装好了"（测试/门禁/最小用例）
- 无隐私/敏感内容——脚本里的 IP、token、个人路径必须先参数化（参考 npu-precision-toolkit 的 env 化处理）

### 第三关：归属判定

| 成果类型 | 归属 |
|---------|------|
| NPU 工程类（排查工具/性能脚本/环境恢复） | `skills/<新 skill>/`（本仓库） |
| Claude 会话类（上下文管理/成本优化） | `.claude/skills/` + `skills/` 薄壳（usage-intelligence 模式） |
| 一次性实验材料（verdict 证据/中间产物） | **不归集**，留 evidence 目录，仅在相关 skill 的 knowledge 里引用 |

## 标准结构（脚手架模板）

```
skills/<skill-name>/
├── SKILL.md          # 必须。frontmatter（name+description 双语触发条件）+ 概述 + 部署 + 快速上手 + 方法论
├── scripts/          # 可执行工具（路径一律 env 化，禁止绝对路径）
├── templates/        # 复用模板（带注释的最小完整示例）
├── knowledge/        # 领域知识（指纹/台账/教训，Markdown）
├── docs/             # 详细文档（可选，README 级）
└── tests/            # 验证（可选但强烈建议）
```

### SKILL.md 写作要点（从 npu-precision-toolkit 提炼）

1. **description 双语**且写清"何时用"+"何时不适用"（与相邻 skill 划界）
2. **部署段**写明运行前提（哪些必须在 NPU 服务器、PC 侧能做什么）
3. **方法论段**收录踩坑教训——这是 skill 区别于纯脚本集的核心价值
4. **归因修正**类知识（被证伪的结论）必须保留并显著标注，防止后人重蹈覆辙
5. 已验证环境表（SoC/CANN/torch/torch_npu 版本）——环境敏感类 skill 必须

## 归集流程

1. **盘点**：列出候选成果（脚本/模板/文档/教训），逐个过三关
2. **参数化**：硬编码路径 → env 变量；敏感信息清除；默认值保持原环境兼容
3. **脚手架**：按标准结构建目录，SKILL.md 按写作要点起草
4. **验证**：`python -m pytorch_yongli_skill --root . list` 能发现新 skill；
   有 tests 的跑 `python -m unittest discover -s tests -q`
5. **登记**：README 的 Integrated skills 列表 + `docs/member-manual.md` 补一行
6. **提交**：git commit（信息格式：`skills: add <name> — <一句话>`）

## 已归集清单（2026-10-09 起）

| skill | 来源项目 | 归集日期 |
|-------|---------|---------|
| npu-precision-toolkit | llama2_7b 精度排查（四根因 + 13 模型穿刺） | 2026-10-09 |
| usage-intelligence（薄壳） | Claude 会话成本优化 | 2026-10-09 |
| server-workspace-hygiene | 服务器整理 + 本地备份（09-28/29 两轮实战） | 2026-10-09 |
| puncture-model-verify（流程层） | 50 模型穿刺计划（13 次执行收敛） | 2026-10-09 |
| red-green-validation | rc2/rc3 verdict + rc4 红灯（三次实战） | 2026-10-09 |

## 待归集候选（性能方向预铺，2026-10-09 勘察）

| 候选 | 状态 | 说明 |
|------|------|------|
| npu-perf-toolkit | ⏳ 等待素材 | 性能方向做出第一个可复用 profile/对比流程时归集（llama 慢 12%、T5 tile-contract 两调查点的产出） |
| gitcode-pr-flow | ⏳ 不够稳定线 | 已执行 2 次（PR-3/#47232、PR-4+6/#47261），判定规则要求 ≥3 次无修改收敛；且含 token 类敏感操作，归集前必须改成"用户提供 token"模式并加安全注意事项。下次提 PR 流程零改动即达标 |
| puncture-model-verify 独立化 | ✅ 已归集（薄壳流程层） | 工具/模板实体留 toolkit，防双份漂移 |

> 候选条目由本 skill 维护：每次归集评审时更新此表，已达标未归集的标 ⏳ 等待，
> 不达标的写清缺什么。
