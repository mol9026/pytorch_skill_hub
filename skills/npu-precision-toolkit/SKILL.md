---
name: npu-precision-toolkit
description: >-
  NPU 图模式（torch.compile triton_experimental 后端）精度排查四件套：
  回归门禁 watchdog（"还对不对"）、金字塔诊断 doctor（"坏在哪层"）、
  kernel 隔离 capture-replay（"定点深挖"）、buffer 复用静态审计（"复用是否危险"）。
  附 13 个模型穿刺 args 模板与缺陷指纹库（llama2_7b 四根因，含 2 个已证伪归因的修正记录）。
  当用户报告 NPU 图模式精度异常（cos 下降、NaN、bitwise 不确定）、需要精度回归门禁、
  或要为新增模型做穿刺验证时使用。
  Use when NPU torch.compile precision issues (cos drop, NaN, nondeterminism),
  precision regression gating, or model puncture verification is needed.
---

# npu-precision-toolkit：NPU 图模式精度排查四件套

## 概述

源自 Llama-2-7b NPU 图模式精度排查项目（2026-09，四根因定位与修复，2 个修复已进
gitcode PR #47232 / #47261）。四工具按"金字塔"分工：

| 工具 | 回答的问题 | 何时用 |
|------|-----------|--------|
| `precision_watchdog.py` | 还对不对？ | 合并前/改配置后跑门禁，exit code 即结论 |
| `npu_compile_doctor.py` | 坏在哪层哪个 kernel？ | 模型输出不对时一键诊断（L1-L4 + 指纹匹配） |
| `kernel_isolate.py` | 这个 kernel 单独跑对不对？ | doctor 定位到站点后 capture-replay 深挖 |
| `buffer_alias_tracker.py` | 产物 buffer 复用危险吗？ | 秒级静态审计编译产物 |

**与 `torch-npu-test-failure-diagnose` 的边界**：那个 skill 是**只读诊断**（给定失败用例
名 + 日志，输出根因分析，不改代码）；本 toolkit 是**主动实验**（在 NPU 服务器上编译、
跑、对拍、重放）。先用那个看日志，需要动手验证时用本 toolkit。

## 部署（必须 NPU 服务器，PC 侧只读结果）

```bash
# 1. 拷贝到服务器
scp -r skills/npu-precision-toolkit/scripts/ <user>@<npu-server>:~/scripts/
scp -r skills/npu-precision-toolkit/templates/puncture_args/ <user>@<npu-server>:~/scripts/puncture_args/

# 2. 环境（每次会话必做）
source <CANN路径>/set_env.sh
conda activate <build环境>
export TORCHINDUCTOR_COMPILE_THREADS=1   # 关键：fork 子进程 NPU 重初始化 bug

# 3. 路径参数化（可选，默认值为项目原始服务器路径）
export NPU_EVIDENCE_DIR=~/evidence/llama2_7b   # 证据输出目录
export NPU_MODEL_DIR=<模型权重目录>             # 真权重快照
export NPU_SCRIPTS_DIR=~/scripts               # args 脚本目录
export NPU_DEBUG_ROOT=~/torch_compile_debug    # 编译产物根
```

## 快速上手

### 场景 A：模型图模式输出不对（诊断）

```bash
# 写 args 脚本（参考 templates/puncture_args/ 下 13 个模板）
# 契约：build_args(device) -> (args_list, ref, model, forward_fn)
python3 scripts/npu_compile_doctor.py --args-script scripts/puncture_args/nanogpt_args.py
# L1 端到端 cos + 确定性（~10min）；cos>0.999 且 bitwise 确定 = HEALTHY
# PROBLEM 时自动进 L2-L4 定位 + 匹配已知缺陷指纹
```

### 场景 B：改了编译器/配置，回归验证

```bash
python3 scripts/precision_watchdog.py --gate g12   # ~10min，四根因回归
python3 scripts/precision_watchdog.py --gate g4    # reduction 竞争专项（10 次 bitwise）
# exit code != 0 即拦截
```

### 场景 C：新模型穿刺（团队 50 模型验证计划）

1. 仿照模板写 `build_args`（随机权重同构即可，无需官方权重）
2. 跑 doctor L1
3. HEALTHY → 按 `knowledge/puncture-ledger.md` 的格式记台账
4. PROBLEM → L2-L4 → 匹配 `knowledge/fingerprints.md` 指纹

### 场景 D：单 kernel 定点深挖

```bash
python3 scripts/kernel_isolate.py capture   # 注入探针重跑，捕获真实 buffer
python3 scripts/kernel_isolate.py test <site> --runs 10   # 重放 10 次看 bitwise
```

## 核心方法论（踩坑沉淀，省 3 天弯路）

1. **L1 是充分的**：cos>0.999 且 bitwise 确定 = 模型健康，无需 L2-L4。深层检查是
   "出了问题后的定位工具"，不是"健康证明工具"。
2. **graph break ≠ 不健康**：dynamo 断图段走 eager（正确性由 eager 保证），
   "HEALTHY"的准确含义是"被捕获的子图全对"。台账需注记（参考 ledger 里
   speech_transformer / llava 的写法）。
3. **cos>1 需注记**：随机权重 + 大 vocab 的 logits 尺度会自然产生 cos 1.00-1.03，
   方向高度对齐即可，不是错误（Llama-3.2-1B 案例 1.032607）。
4. **多根因同时修复后，每个修复必须单独隔离验证**（红绿灯：单撤一个 → 跑门禁 →
   对照基线）——本项目两个"修复"（rc2/rc3）就是没做隔离验证，后续 verdict 实验
   才发现是归因错误，白白走了 PR 流程。见 `knowledge/PATCHES.md` #7 行。
5. **kernel 重放一律用 capture-replay**，不要手工构造输入——手工构造的 buffer 布局
   与图内实际布局不一致，会得到假阴性（2026-09-28 PR-3 材料教训）。
6. **缓存不感知 monkeypatch**：改了 in-place 补丁必须 `rm -rf /tmp/torchinductor_root`
   再跑，否则读旧产物。

## 缺陷指纹库（`knowledge/fingerprints.md`）

四根因 + 通用模式，每条含"特征 → 假设 → 验证建议"：

| 指纹 | 特征 | 根因 |
|------|------|------|
| 相位漂移 | 误差随位置单调放大 | inv_freq 量化类（已证伪为噪声级，见归因修正） |
| 索引/轴合并错位 | kernel 单点重放 ≠ 图内 | 跨轴合并（PR #47232 已修） |
| 并发竞争 | 同输入多次执行 bitwise 不一致 | in-place reduction（PR #47261 已修） |
| layout 契约缺失 | strided 输入错 / contig 对 | aclnn 类算子（**注意：FA v3 已实测支持 strided，勿重蹈 rc2 覆辙**） |
| buffer 复用时序 | 单点全对、整链错 | in-place 污染（PR #47261 变体） |

## 穿刺台账（`knowledge/puncture-ledger.md`）

13 模型全绿的记录 + 每模型注记 + 架构覆盖预判表 + L2-L4 工具边界说明。
新模型穿刺后按同格式追加。

## 已验证环境

Ascend 910B2 / CANN 9.1.0 / torch 2.14.0.dev20260801+cpu / torch_npu 2.14.0.dev20260801+git1b64dad /
transformers 5.17.0 / Python 3.12
