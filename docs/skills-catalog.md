# Skill 能力目录（按"想做什么"选）

> 总分式手册：每节先一句话定位，再给触发场景、前置条件、核心能力、边界。
> 详细协议看各 skill 的 SKILL.md；本文回答"我该用哪个"。

## 快速选型表

| 你想做什么 | 用哪个 skill | 一句话理由 |
|-----------|-------------|-----------|
| 检视一个 PR | `pr-review` | 结构化代码检视 |
| 重构/解耦 PyTorch 测试用例 | `pytorch-test-refactoring` | agent 编排的全流程重构 |
| 判断本地测试改动是否合理 | `test-modification-advisor` | 修改建议与覆盖保持检查 |
| NPU 用例挂了，想知道根因 | `torch-npu-test-failure-diagnose` | 只读日志分析，定位 torch_npu/CANN/上游 |
| 模型图模式输出不对（cos 降/NaN/不确定） | `torch-npu-test-failure-diagnose` → `npu-precision-toolkit` | 先读日志，再跑实验验证 |
| 上线前确认精度没退化 | `npu-precision-toolkit`（watchdog 门禁） | exit code 即门禁 |
| 验证一个新模型的图模式精度 | `puncture-model-verify` | 穿刺标准协议 + 13 个模板 |
| 一个修复要定案/提 PR | `red-green-validation` | 隔离红绿灯，防止"修复不存在的问题"（rc2/rc3 教训） |
| 服务器乱了/要备份/阶段收尾 | `server-workspace-hygiene` | 巡检→归档→备份→同步四步 |
| 工作中沉淀了脚本/模板想共享 | `skill-harvest` | 三关判定 + 标准脚手架 |
| token 消耗异常/想知道模型怎么路由的 | `model-switch-guard` | 网关日志审计 + 缓存损失估算 |
| 会话上下文管理/压缩提醒 | `usage-intelligence` | hooks 自动检测 + 收益账本 |

---

## 测试质量类

### pr-review

- **定位**：PR 代码检视 skill（文档型，来自社区）。
- **触发**：用户要求检视 PR / review 代码变更。
- **使用**：`python -m pytorch_yongli_skill --root . read pr-review` 查看完整检视协议。

### pytorch-test-refactoring

- **定位**：PyTorch 测试用例解耦与重构的 agent 编排流水线（analyst → coder → checker → reviewer 多角色 harness）。
- **前置**：可执行 skill，CLI `run` 直连其 `orchestrator.py`；详见其 README.md 与 AGENTS.md。
- **边界**：面向测试文件的解耦重构，不做 NPU 精度验证。

### test-modification-advisor

- **定位**：本地测试修改的合理性建议（修改前评估 + 覆盖保持检查）。
- **前置**：可执行脚本在 `scripts/`；协调指南与模板齐备。
- **边界**：只建议不改码；与 `pr-review` 的区别是面向"改测试的人"而非"审代码的人"。

### torch-npu-test-failure-diagnose

- **定位**：NPU 用例失败的**只读**根因分析——分类失败、判断可修复性（torch_npu 侧 / CANN 侧 / 上游）、输出结构化诊断报告。
- **输入**：用例名 + 失败日志（examples/ 有 3 个完整示例）。
- **核心能力**：knowledge/ 下含分发缺失、不支持模式、分布式陷阱、上游 issue 四类知识库。
- **边界**：**不执行任何实验、不改代码**。需要动手验证 → 转 `npu-precision-toolkit`。

---

## NPU 图模式精度类

> 这三个 skill 同源（llama2_7b 精度排查项目，2026-09），按"工具 → 流程 → 定案协议"分层。
> 共同前提：NPU 服务器 + 环境（CANN set_env.sh / conda build / TORCHINDUCTOR_COMPILE_THREADS=1）。

### npu-precision-toolkit（工具层）

- **定位**：四件套可执行工具 + 模板 + 知识库。
- **核心能力**：
  - `precision_watchdog.py`：精度回归门禁（G1-G4，exit code 即结论，~10min）
  - `npu_compile_doctor.py`：金字塔诊断 L1-L4（端到端 cos → kernel 隔离 → buffer 溯源 → 指纹匹配）
  - `kernel_isolate.py`：单 kernel capture-replay（10 次 bitwise 判确定性）
  - `buffer_alias_tracker.py`：编译产物 buffer 复用静态审计（秒级）
  - `templates/puncture_args/`：13 个模型的 build_args 模板
  - `knowledge/fingerprints.md`：缺陷指纹库（**含 2 个归因修正的显著标注**，防止重蹈 rc2/rc3 覆辙）
- **边界**：需要真机编译执行；PC 侧只能读证据。所有路径已 env 化（`NPU_EVIDENCE_DIR` 等）。

### puncture-model-verify（流程层）

- **定位**：新模型穿刺验证的**标准协议**——工具在 toolkit，本 skill 定义怎么用。
- **核心能力**：build_args 四元组契约（`build_args(device) -> (args, ref, model, forward_fn)`）、读数三查（cos/确定性/cos>1 注记）、绕依赖三板斧（from_config 同构/子模块直构/手写对照）、台账格式。
- **边界**：不重复工具文档；HEALTHY 的判定语义（graph break / cos>1 注记）以本 skill 为准。

### red-green-validation（定案协议层）

- **定位**：任何修复**定案或提 PR 前**的隔离验证协议。
- **核心能力**：五步（基线 → 单撤 → 红灯 → 对照判定 → verdict JSON 落盘）+ 撤销铁律（全触点枚举、清 inductor 缓存）+ 三条判定细则（环境核查先行、产物层取证、历史证据无脚本时的处理）。
- **何时强制**：多修复同时生效时（本项目 rc2/rc3 就是因为没做隔离验证走了弯路）。

---

## 工程流程类

### server-workspace-hygiene

- **定位**：NPU 排查服务器的整理与备份标准流程（两轮实战收敛）。
- **核心能力**：四步——巡检（du 排行 + 归属确认）→ 归档（debug run 按时间窗 tar）→ 备份（带日期 tar + md5 双端校验 + 排除清单）→ 同步（PATCHES/台账双端一致）。
- **踩坑沉淀**：evidence 只增不删、/tmp 会被服务器清理、共享服务器先确认归属、tar 前先 du 估算。

### skill-harvest

- **定位**：判断工作沉淀是否值得收编进本仓库，并按标准结构生成 skill 草稿。
- **核心能力**：三关判定（价值：重复≥3/有契约/踩坑/跨人复用 → 稳定：收敛/可验证/无敏感 → 归属：NPU 类/会话类/不归集）、标准目录脚手架、SKILL.md 写作五要点、候选清单维护。
- **使用**：说「归集」或 usage-intelligence 发出 skill 化提醒时走此流程。

---

## 会话与成本类

### model-switch-guard

- **定位**：从 CC Switch 网关日志审计**真实的模型路由变化**，区分四类事件，估算切换的缓存损失。
- **前置**：CC Switch 启用 local proxy（数据库 `~/.cc-switch/cc-switch.db`）。
- **核心能力**：单一观测点设计（不监听 UI/命令，只看实际路由结果）；PROVIDER_SWITCH / RETRY_FALLBACK / GATEWAY_JITTER / SUBAGENT 四分类，0 误报回归（23 天 4256 请求）；`report [days]` 输出风险上下文总量。
- **边界**：依赖 CC Switch 日志格式；其他网关需改数据源适配层。

### usage-intelligence（薄壳）

- **定位**：Claude Code 会话用量与成本优化（上下文监控、压缩时机提醒、收益账本、skill 化识别）。
- **部署**：**不在本仓库运行**——hooks 必须装在各自项目的 `.claude/` 下；安装三步法见其 SKILL.md。
- **核心能力**：「写状态」固化线索（压缩/换会话零损失）、「查收益」账本报表、「封装」触发 skill-harvest。

---

## 版本与维护

- 归集状态与候选清单：见 `skills/skill-harvest/SKILL.md` §已归集清单 / §待归集候选
- 新增 skill 必须走 skill-harvest 流程（三关判定不过不收）
- 本目录由 skill-harvest 维护：每次归集/退役后同步更新
