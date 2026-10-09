# precision-watchdog（提案 C 交付）—— NPU 图模式精度回归门禁

> 一条命令跑三层门禁，防止已修复的四类精度缺陷在任何 torch_npu / triton-ascend 变更中复发。
> 退出码即门禁：`0`=通过 `1`=精度/确定性失败 `2`=基础设施错误。

## 为什么存在

2026-09 三天内定位并修复了 Llama-2-7b NPU 图模式 cos=0.75 的四个独立根因（跨轴合并索引错位 / FA layout 契约缺失 / fp16 inv_freq 相位漂移 / in-place reduction 跨核竞争）。全部修复已验证（整模型 cos=1.000046、bitwise 确定性恢复），但这些修复分布在 site-packages 的 monkeypatch 里——**后续任何 torch_npu/triton-ascend 升级都可能悄悄冲掉它们**。watchdog 把"精度还达标吗"变成一条命令可回答的问题。

## 四层门禁

| 层 | 抓什么 | 判据 | 耗时（清缓存后） |
|----|--------|------|------------------|
| G1 单层精度 | 索引错位类（快速信号） | 1 层随机权重 Llama graph vs eager cos > 0.999 | ~4 min |
| G2 执行确定性 | 并发竞争类 | 同 compiled 模型 10 次执行 bitwise 一致 | （与 G1 同编译）~1 min |
| G3 端到端精度 | 团队口径 + 构造侧语义 | TorchBench llama_v2_7b_16h cos > 0.999 | ~10 min |
| G4 真实权重单层 | **编译器侧回归主门禁**（历史故障主路径） | from_pretrained 真权重 1 层 cos > 0.999 | ~5 min |

用法：
```bash
# 环境（每次必须）
source /home/z00599499/envs/cann/9.1.0/cann-9.1.0/set_env.sh
conda activate /home/z00599499/envs/miniconda/envs/build
cd /home/z00599499

python3 scripts/precision_watchdog.py --gate all --json /home/z00599499/watchdog_results/llama2_7b/watchdog_report.json
# 或分层：--gate g12 / --gate g3 / --gate g4
```

## 红绿灯验证结果（2026-09-24，本工具自身可信度的证据）

### 绿灯（修复态基线）

| 门禁 | 结果 | 数值 |
|------|------|------|
| G1 | PASS | cos=1.000004 |
| G2 | PASS | 10 次 bitwise 一致（maxdiff=0.0） |
| G3 | PASS | cos=1.004134（与 2026-09-23 基线复现） |
| G4 | PASS | cos=1.000014（v1.2，inv_freq f32 语义修正后；v1.1 为 1.000016） |

### 红灯（回退修复，验证门禁真的会拦截）

| 回退组合 | G1 | G2 | G3 | 判定 |
|----------|-----|-----|-----|------|
| 撤 fix2+4（layout tag + inplace 保护） | PASS | PASS | PASS（缓存内） | **盲区：该 shape 不触发** |
| 撤 fix1（can_merge_dims 守卫） | PASS | PASS | PASS* | *产物确实变（非相邻合并发生）但结果正确——该图里可合并对不含 n/s 交叉 |
| 撤 fix1+2+4（完整修复前状态，清缓存 616s 真编译） | — | — | **PASS cos=1.004134** | **决定性发现：G3 抓不到编译器侧回归** |

**决定性红灯的结论**（2026-09-24 14:53，616s 清缓存全量编译）：清空 /tmp/torchinductor_root 确认真实重编译、rc1+2+4 全部回退，G3 依然与绿灯逐位一致。机理：TorchBench llama_16h 的历史 0.657 主要由**根因 3（构造侧 inv_freq 量化）**贡献，watchdog 按正确语义构造（inv_freq f32）后，该路径下编译器侧三修复不改变输出；而真实权重路径的历史 0.745（1 层 layer0 cos=0.9177）是纯编译器缺陷。

**因此补上第四层门禁 G4**（真实权重 from_pretrained 单层，历史故障主路径）。修正后的覆盖矩阵：

| 根因 | 捕获门禁 | 说明 |
|------|----------|------|
| 1 跨轴合并索引 | **G4** | 真权重单层是 rc1 的显形场（历史 0.9177） |
| 2 FA layout 契约 | G4（叠加）+ G3 端到端 | |
| 3 inv_freq 相位漂移 | **G3**（构造侧正确语义下测编译栈）+ watchdog 脚本内固化 | |
| 4 in-place 并发竞争 | G2 + G4 | 历史上 NaN 出现在真权重整模型 |

### G4 红灯（全回退态，预期 FAIL → 实际 PASS，重要发现）

```
状态：rc1+2+4 全撤 + 清缓存 + 315s 真实编译（非缓存命中）
预期：cos≈0.92（历史 minimal_repro 同构造 1 层 layer0 = 0.9177）
实际：cos=1.000016 → PASS
```

**结论：历史坏合并没有在今天的编译中形成。** 取证过程（逐步排除）：

1. ~~缓存命中~~ —— 已清 /tmp/torchinductor_root，编译 315s/616s 是真实时长
2. ~~G4 构造与历史不同~~ —— 与 minimal_repro 完全同构（from_pretrained、1 层、seq=128、seed 一致）
3. **✓ 回退不完整（最终确认）**：对 `backup_triton_codegen.py`（369KB）做内容比对，锁定其对应 `triton_experimental/codegen/triton.py`，**live 比 backup 多 13 行——`node.decide_inplace_update()` 被替换为注释 + `pass`，这是根因 4 的第二修复点**。之前的"全回退"只撤了 monkeypatch（loader 调用），codegen 侧的 pass 仍在生效，rc4 实际未回退。

**完整在位修改清单（6 个，修正 PR 计划的"5 个"）**：

| # | 文件 | 内容 | 根因 |
|---|------|------|------|
| 1 | `triton/backends/ascend/utils.py:438` | c++17→c++20 | 基础设施 |
| 2 | `torch_npu/_inductor/select_algorithm.py` | `**kwargs` 签名 | 基础设施 |
| 3 | `torch_npu/_inductor/triton_experimental/fx_passes.py:479` | can_merge_dims 相邻轴守卫 | rc1 |
| 4 | `torch_npu/_inductor/triton_experimental/__init__.py` | `_install_fa_layout_constraints`（FA tag + inplace monkeypatch） | rc2+rc4 |
| 5 | `torch_npu/_inductor/__init__.py:383` | loader 调用 | 修复 4 的安装点 |
| 6 | `triton_experimental/codegen/triton.py:7718` | in-place 调用点 pass 化 | **rc4 第二点** |

**方法论教训（写进案例分享值得）**：红灯测试的回退必须枚举全部修复触点——rc4 有 codegen 调用点和 monkeypatch 两层，只撤一层测出的是"假阴性"。backup 文件与 live 的逐一 diff 是枚举触点的可靠手段。

**这一发现改变了 watchdog 的使用姿势**（见下节）。G4 的正确性验证来自另一面：它精确复现了修复态基线（绿灯 1.000016），且历史 0.9177 证明它测的路径就是故障主路径。

### 真完整回退红灯（6 修改点全撤，决定性实验）—— ✅ 门禁抓到

```
状态：#3 守卫注释 + #4/#5 loader 调用撤 + #6 codegen pass 回退为 node.decide_inplace_update()
      + 清 /tmp/torchinductor_root + 340s 真实全量编译
预期：G4 cos 显著 < 0.999（历史同构复现为 0.9177）→ REAL_EXIT=1
实际：G4 cos = None（NaN）→ FAIL，REAL_EXIT=1  ✅
```

**门禁验证闭环完成**：绿灯（修复态）全过 + 红灯（完整回退 6 修复点）exit=1 拦截。NaN 正是根因 4（in-place reduction 跨核竞争）的指纹——与历史"偶发 NaN 且每次结果不同"完全吻合。同时反向确认了两件事：

1. **之前红灯"不复现"的谜底就是回退不完整**：rc4 有 monkeypatch 和 codegen 调用点两层修复，只撤 monkeypatch 时 codegen 的 pass 仍在拦截 in-place，故障无法显形。backup↔live 逐一 diff 是发现遗漏修复点的可靠手段。
2. **修复的必要性铁证**：6 修复点全撤的当下编译立刻产生 NaN——修复不是"锦上添花"，是当前图模式在该路径上正确性的必要条件。

**对 CI 的最终意义**：watchdog 的红灯不是理论推演——它真的在 340 秒内把一个"看起来能跑"的坏编译拦在门外（exit code 1 + JSON 报告可直接作 PR 证据）。

## 对 CI 策略的最终建议

| 场景 | 跑什么 | 理由 |
|------|--------|------|
| torch_npu/triton-ascend 合并前（~10min） | `--gate g12 + g4` | G1/G2 快速信号 + G4 历史故障主路径 |
| nightly（~20min） | `--gate all` 全四层 | G3 团队口径端到端 |
| 大版本 / 升级 torch（含清缓存） | `--gate all` + `rm -rf /tmp/torchinductor_root` | 缓存不感知 monkeypatch，跨版本必须清 |
| 高风险 pass 改动 | g4 重复 3-5 次 | **非确定性缺陷单次通过无效**，需多次采样 |

## 已知边界

1. **缓存感知**：FX graph cache / kernel cache 的键不感知 site-packages monkeypatch。做对比实验时必须 `rm -rf /tmp/torchinductor_root`；日常回归不受影响（代码变了缓存键就变）。
2. **非确定性缺陷**：rc1 类坏合并依赖编译期调度决策，单次跑 g4 通过不能证明无回归——高风险变更建议重复采样（g4 跑 3 次）。
3. **根因 3 的位置**：inv_freq 相位漂移是构造侧问题（from_config+.to(fp16) vs from_pretrained），watchdog 在 G1/G3 内已按正确语义构造。若上游修了 transformers 的 fp16 转换，G3 数值会小幅变化，属预期。
4. **G4 依赖本地权重**：需 `/home/z00599499/llama-2-7b/...` 存在（内网机可改为环境变量指定路径）。

## 接入 CI 的最小方案

```bash
# torch_npu / triton-ascend 合并前
python3 scripts/precision_watchdog.py --gate g12 --json $ARTIFACT/watchdog_g12.json   # ~5 min
# nightly
python3 scripts/precision_watchdog.py --gate all --json $ARTIFACT/watchdog_nightly.json
```
exit code != 0 即拦截。JSON 报告含逐门禁数值，可直接作为 PR 附件。

## 文件位置

| 文件 | 位置 |
|------|------|
| 主程序 | `<NPU 服务器>/scripts/precision_watchdog.py`（本仓库 `skills/npu-precision-toolkit/scripts/`） |
| 历史基线 | `/home/z00599499/evidence/llama2_7b/torchbench_llama16h_stats.json` |
| 红绿灯 JSON | `/home/z00599499/watchdog_results/llama2_7b/watchdog_*.json` |
| 四根因完整证据链 | `/home/z00599499/docs/bug定位报告_triton_experimental数值错误.md`（v5 §11.4 PR 方案） |

