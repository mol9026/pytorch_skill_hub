# npu-compile-doctor（提案 A 交付）—— NPU 图模式精度问题自动诊断

> **一条命令执行金字塔排查**：输入"图模式输出不对"的模型，354 秒内自动完成
> 层级排除 → kernel 隔离 → buffer 溯源 → 特征指纹匹配，输出"哪一层、哪个 kernel 站点、
> 像哪类已知缺陷、怎么验证"的诊断报告。
>
> 提案关系：A = C（watchdog 门禁）+ B（kernel 隔离）+ L3/L4 编排 + 指纹库。

## 用法

```bash
# 环境（每次必须）
source /home/z00599499/envs/cann/9.1.0/cann-9.1.0/set_env.sh
conda activate /home/z00599499/envs/miniconda/envs/build
cd /home/z00599499

# 内置：真权重单层 Llama（快速自检，23s）
python3 scripts/npu_compile_doctor.py

# 任意模型：提供 build_args(device)->(args, ref, model) 脚本
python3 scripts/npu_compile_doctor.py --args-script my_model.py

# 只跑部分层（L1 健康 23 秒短路；全层 ~6-10 分钟含编译）
python3 scripts/npu_compile_doctor.py --levels 1,4
```

退出码：`0`=图模式健康 `1`=检出问题（含指纹）`2`=基础设施错误。

## 四层诊断

| 层 | 干什么 | 耗时（有缓存产物时） | 产出 |
|----|--------|---------------------|------|
| L1 层级排除 | graph vs eager 整体 cos + 两次执行确定性；healthy 短路 | ~23s | 是否继续 + 非确定性初判 |
| L2 kernel 隔离 | 注入探针跑真实 Runner.call，逐站点重放 3 次 | ~20s | RACE / REPLAY-MISMATCH / OK 三态逐站点 |
| L3 逐 buffer 溯源 | 跨站点 data_ptr 同址改写检测（in-place 污染） | ~2s | 污染链证据 |
| L4 特征指纹 | 误差按位置分布分析（前/后 1/4 对比） | ~2s | 相位类 / NaN 爆炸类指纹 |

## 缺陷指纹库（可持续积累）

| 指纹 key | 指向 | 已知案例 |
|----------|------|----------|
| `nondeterministic` | 并发竞争（in-place 读改写类） | llama2_7b 根因4 |
| `err_grows_with_position` | 相位漂移（inv_freq 量化类） | llama2_7b 根因3 |
| `kernel_replay_mismatch` | 索引/轴合并错位类 | llama2_7b 根因1 |
| `downstream_break_after_strided_input` | layout 契约缺失类 | llama2_7b 根因2 |
| `single_ok_chain_bad` | buffer 复用/调度时序类 | llama2_7b 根因4 变体 |

每条含"假设 + 验证建议"，doctor 匹配后直接打印下一步动作。新根因发现后往 `FINGERPRINTS` 加一条即可（脚本内自包含）。

## 红绿灯验证结果（2026-09-24）

### 绿灯（修复态，全过）

| 层 | 结果 |
|----|------|
| L1 | cos=1.000014 且 bitwise 确定 → HEALTHY 短路 exit=0（23s） |
| L2（--levels 2,3,4 显式跑） | 14 站点全 OK |
| L3 | 无跨站点污染 |
| L4 | 误差均匀（无位置放大） |

### 红灯（rc1+2+4 全撤 + 清缓存，组合故障）

```
[L1] graph vs eager cos = nan；两次执行 bitwise 不一致 → 并发竞争类指纹
[L2] site 12（最终 RMSNorm 融合 kernel）重放不确定 → RACE
[L4] 前1/4 误差 inf vs 后1/4 nan → 数值爆炸指纹
匹配: 并发竞争（in-place 读改写类）→ llama2_7b 根因4
VERDICT: PROBLEM FOUND，exit=1（354.9s，含 ~5.5min 编译）
```

**当年 3 天的手工排查，doctor 6 分钟跑完并给出同样结论**（L2 定位的 site 12 正是根因 4 的案发 kernel——最终 RMSNorm）。

### 红灯过程中的有效发现

单根因 3（fp16 inv_freq）在修复态编译器上**无法独立复现**（4 层 seq=512 实测 cos=1.00014）——
历史 0.657 是"根因3 + 修复前编译器（根因1/2/4）"的叠加效应。这与案例分享记录的
"四根因耦合"一致：单撤任一修复无法复现历史故障，红灯测试需用组合故障。此发现已反哺
watchdog 的 CI 建议（高风险变更需组合回归 + 多次采样）。

## 已知边界

1. **L1 健康即短路**是特性（快速自检 23s）；完整体检用 `--levels 2,3,4`
2. **L2/L3 依赖产物**：需要一次成功的 compile（doctor 自身会触发，或复用 torch_compile_debug 里最新 run）
3. **缓存不感知 monkeypatch**：红灯/对比实验前 `rm -rf /tmp/torchinductor_root`（同 watchdog 边界 1）
4. **指纹库初版 5 条**：均来自 llama2_7b 案例；新模型新根因出现时持续追加
5. **`--args-script` 契约**：`build_args(device) -> (args_list, ref_output, model)`，args 顺序须与产物 Runner.call 形参一致（参考 `scripts/kernel_isolate.py:build_llama1layer_args`）

## 文件位置

| 文件 | 位置 |
|------|------|
| 主程序 | `<NPU 服务器>/scripts/npu_compile_doctor.py`（本仓库 `skills/npu-precision-toolkit/scripts/`） |
| 故障注入样例 | `/home/z00599499/scripts/fault_fp16_invfreq.py`（根因3 语义，红灯实验用） |
| 诊断报告 | `/home/z00599499/evidence/llama2_7b/npu_doctor_report.json` |
| 依赖：提案 B | `/home/z00599499/scripts/kernel_isolate.py`（L2 层引擎） |
| 依赖：提案 C | `/home/z00599499/scripts/precision_watchdog.py`（门禁，doctor 与之互补） |

## 三工具协同图

```
日常开发 ──> watchdog（每次合并前，10-20min，"还对不对"）
出了问题 ──> doctor（一次诊断，6-10min，"坏在哪一层哪个 kernel、像什么"）
       └──> kernel_isolate（定点深挖，秒级重放，"这个 kernel 到底什么行为"）
```
