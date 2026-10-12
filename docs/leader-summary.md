# PyTorch skill 整合方案汇报材料

## 1. 背景

项目中已经汇集了多个相互独立的 PyTorch 相关 skill（现 11 个，四大类），分别覆盖：

- 测试质量类：PR 检视 `pr-review`、用例解耦与重构 `pytorch-test-refactoring`、测试修改建议 `test-modification-advisor`、NPU 用例失败诊断 `torch-npu-test-failure-diagnose`
- NPU 图模式精度类（llama2_7b 项目沉淀）：四件套工具 `npu-precision-toolkit`、穿刺协议 `puncture-model-verify`、修复隔离验证 `red-green-validation`
- 工程流程类：服务器整理备份 `server-workspace-hygiene`、成果归集 `skill-harvest`
- 会话与成本类：模型路由审计 `model-switch-guard`、会话成本优化 `usage-intelligence`

这些 skill 各自具备明确职责，但此前分散在不同目录中，缺乏统一入口与发现机制，导致：

- 任务定位不一致
- 资源复用不足
- 维护成本偏高
- 新成员上手难度增加

## 2. 整合目标

将多个 skill 统一收敛为一个可发现、可检索、可调度的总入口，形成如下能力：

- 统一发现所有 skill
- 统一展示 skill 元数据
- 统一读取 skill 说明文档
- 统一执行可运行 entrypoint
- 自动按场景选择最合适 skill

## 3. 已形成的能力

当前已完成的整合层包括：

- `skills/` 目录中的原始 skill 保持独立，不强制改造原逻辑
- 统一 registry 自动扫描 `SKILL.md` 进行发现
- CLI 入口统一管理全部 skill
- 支持 list / show / read / run / route 等统一操作
- 支持自动路由用户诉求到对应 skill

## 4. 实际效果

现在团队可以通过统一入口执行以下动作：

```bash
python -m pytorch_yongli_skill --root . list
python -m pytorch_yongli_skill --root . show pr-review
python -m pytorch_yongli_skill --root . read pr-review
python -m pytorch_yongli_skill --root . run pytorch-test-refactoring -- --help
python -m pytorch_yongli_skill --root . route "我需要诊断 PyTorch-NPU 用例失败"
```

这意味着：

- 不需要逐个记住目录位置
- 不需要人为维护各 skill 映射表
- 能用统一入口进行索引、阅读和调用
- 后续追加新 skill 时，扩展成本大幅下降

## 5. 价值

本方案的核心价值体现为：

1. 统一入口降低协作门槛
2. 保持各 skill 独立，减少互相耦合
3. 提升扩展性，便于新增成员技能接入
4. 提升项目治理效率，便于统一管理与演进
5. 为后续进一步做“自动选择与工作流编排”打好基础

## 6. 当前阶段判断

该整合已完成“统一发现 + 统一入口 + 自动路由”的基础闭环，并完成两批实战归集（NPU 精度工具 + 工程流程协议，能力总览见 `docs/skills-catalog.md`），适合作为统一 skill 平台的基础设施。

后续如果继续增强，可以进一步扩展：

- 更细粒度的意图分类
- 多 skill 工作流编排
- 自动优先级选择
- 任务执行上下文管理

## 7. 结论

本次整合不仅解决了技能分散带来的使用问题，也为后续形成一个统一的 PyTorch skill 生态奠定了基础，提升了项目的可维护性和协作效率。
