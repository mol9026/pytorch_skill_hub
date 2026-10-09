# NPU 图模式缺陷指纹库

> 源自 llama2_7b 四根因排查（2026-09）+ 两个归因修正（2026-09-29 / 10-09）。
> 用法：doctor L2-L4 产出特征 → 查表定位 → 按"验证建议"执行确认实验。
> 每条指纹的"特征"对应 `npu_compile_doctor.py` 的 fingerprint 字段值。

## 指纹总表

### 1. 相位漂移（inv_freq 量化类）

- **匹配特征**：`err_grows_with_position`（误差按位置分桶严格单调递增）
- **假设**：构造侧把 fp32 常量（inv_freq 等）量化成模型 dtype，相位误差 × 位置线性放大
- **验证**：检查 inv_freq.dtype；对照 from_pretrained（保 f32）vs from_config+.to(fp16)（会量化）
- **⚠ 归因修正（2026-09-29）**：transformers 5.17.0 的 `LlamaRotaryEmbedding.forward`
  已将 inv_freq cast 回 fp32 计算；量化构造下实测无差异（graph cos 1.004152 vs f32
  构造 1.004134）。**此指纹在修复态编译器上为噪声级**——命中时先跑对照实验再下结论，
  勿直接按此修复。证据：`evidence/llama2_7b/pr5_verdict.json`

### 2. 索引/轴合并错位类

- **匹配特征**：`kernel_replay_mismatch`（单点重放与图内不一致）
- **假设**：融合 kernel 的索引按错误 layout 生成（跨轴合并/重排）
- **验证**：`kernel_isolate.py test <site>`：喂 contiguous 输入对照；查 fx_passes 轴合并
- **根因实例**：`can_merge_dims` 允许不相邻轴合并，`reindex()` 位置语义无法表达 →
  RoPE kernel n/s 系数交叉。**已修复**（gitcode PR #47232，相邻轴守卫）
- **实锤手法**：kernel1 喂 BNSD-contiguous 输入 bitwise 正确、喂 mm 行主序输入全错——
  两行代码对照把"数学错"和"索引 layout 错"一刀分开

### 3. 并发竞争（in-place 读改写类）

- **匹配特征**：`nondeterministic`（同输入多次执行 bitwise 不一致）
- **假设**：reduction kernel 读改写同一 buffer，48 核 group dispatch 下循环交错
- **验证**：`kernel_isolate.py test <site> --runs 10` 看 bitwise；查产物
  `mutated_arg_names: ['in_out_ptr0']` + `'reuse'` 注释
- **根因实例**：RMSNorm 融合 kernel 统计循环/写回循环共用 in_out buffer，分母用脏数据，
  输出放大 ~300×，偶发 NaN。**已修复**（gitcode PR #47261，reduction 节点跳过 in-place）

### 4. layout 契约缺失类

- **匹配特征**：`downstream_break_after_strided_input`（第一个错的 buffer 上游是
  喂 strided 数据的 extern op）
- **假设**：aclnn 类算子要求物理 contiguous 但 op 未声明 tag，inductor 直喂 strided 数据
- **验证**：喂 contiguous 对照（strided 错 / contig 对 = 实锤）
- **⚠ 归因修正（2026-10-09）**：aclnn FA v3 **实际支持 strided BNSD**（D 维 unit stride
  即可）——隔离实验三组全绿（前向直调 / 反向 grad / llama16h 全模型），历史 strided
  cos=0.158 判定为 rc1 未修复期的测量污染。**勿对 FA v3 直接套用此指纹**；其他 aclnn
  算子命中时仍按验证流程确认。证据：`evidence/llama2_7b/rc2_verdict.json`

### 5. buffer 复用/调度时序类

- **匹配特征**：`single_ok_chain_bad`（所有 kernel 单点正确但整链错）
- **假设**：in-place 复用把后续输出写进了仍被读的 buffer
- **验证**：doctor L3 跨站点污染检测（data_ptr 同址比对）；查产物 `bufN = bufM` +
  `del bufM  # reuse` 模式
- **静态预检**：`buffer_alias_tracker.py`（秒级，标记 reduction+inplace 组合）

## 两个归因修正的共同教训

rc2 / rc3 都是在"多根因同时修复、全绿"之后才发现归因错误的——**绿灯 ≠ 每个修复
都有必要**。定案任何修复前必须做隔离红绿灯：单撤该修复 → 跑门禁 → 与全修复态基线
对照。流程见 SKILL.md 方法论第 4 条。

## 环境指纹（排查先核对）

| 项 | 已验证值 |
|----|---------|
| SoC | Ascend 910B2 |
| CANN | 9.1.0 |
| torch | 2.14.0.dev20260801+cpu |
| torch_npu | 2.14.0.dev20260801+git1b64dad |
| 已知坑 | `TORCHINDUCTOR_COMPILE_THREADS=1` 必须 shell export（fork 子进程 NPU 重初始化）；缓存不感知 monkeypatch，改补丁必清 `/tmp/torchinductor_root` |
