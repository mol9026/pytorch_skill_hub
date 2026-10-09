# kernel-isolation-kit（提案 B 交付）—— NPU 融合 kernel 单点隔离执行器

> 把"从编译产物抽出单个 kernel、用真实 buffer 单独执行对拍"从手工 30 分钟变成一条命令。
> 这是金字塔排查法 L2 层（kernel 级隔离）的工具化，也是提案 A（npu-compile-doctor）的组件。

## 为什么存在

2026-09 的 Llama-2-7b 排查中，"单个 kernel 抽出来对不对"是最频繁的一步（~40 个实验里近半数涉及），
但每次手工做都要：读懂 output_code.py 的 Runner.call → 找到 kernel 调用点 → 从注释还原 buffer 语义 →
手工构造输入 → 处理 stream 参数 → 写对拍逻辑。门槛高且易错（本次排查中两次重大误导都来自
手工脚手架传参错序）。本工具用 **capture-replay 架构**消掉全部手工步骤：

```
list     列出产物全部 kernel 调用点（形状 + 源节点注释，30 秒看懂整图结构）
capture  注入探针 → 真实模型跑一遍 → 抓取每个调用点的 pre/post buffer（含仪器自检）
test     任意 kernel 用捕获的真实输入重放 N 次：
         - N 次间 bitwise 不一致   → RACE（并发竞争指纹，根因4类）
         - 一致但 ≠ 捕获 post      → REPLAY-MISMATCH（输入状态/layout 依赖）
         - 全一致                  → KERNEL OK（单点正确，错误在链路组合/上游 → 去 L3 逐 buffer 对拍）
```

## 用法

```bash
# 环境（每次必须）
source /home/z00599499/envs/cann/9.1.0/cann-9.1.0/set_env.sh
conda activate /home/z00599499/envs/miniconda/envs/build
cd /home/z00599499

# 1. 看结构（30s，无需 NPU 计算）
python3 scripts/kernel_isolate.py list

# 2. 抓取真实 buffer（~4min，llama 1层真权重默认；任意模型用 --args-script）
python3 scripts/kernel_isolate.py capture

# 3. 隔离测试（秒-分钟级）
python3 scripts/kernel_isolate.py test --kernel 13 --runs 10   # site 序号
python3 scripts/kernel_isolate.py test --kernel rmsnorm        # 名称片段

# 任意模型/任意产物：
python3 scripts/kernel_isolate.py capture --artifact <run_dir> --args-script my_args.py
```

`my_args.py` 只需提供 `build_args(device) -> (args_list, eager_ref, model)`，
args 顺序 = Runner.call 形参顺序（看产物源码第一行解包）。

## 验证记录（2026-09-24）

| 检验 | 结果 |
|------|------|
| list：最新产物 14 调用点解析 | ✅ 形状 + 源节点注释齐全（RMSNorm/RoPE/FA 族/最终 RMSNorm 一眼可辨） |
| capture：仪器自检 runner vs eager | ✅ cos=0.999714 > 0.999（args 顺序正确的证明） |
| test：全部 14 site（3-10 次重放） | ✅ 全部 bitwise 一致 + replay==capture（修复态整链体检全绿） |
| 历史有效性（RACE 判定指纹） | kernel1_verdict/iso 系列实验曾用同方法抓到 10 次 bitwise 不一致 → 定位根因 4 |

## 判定与四根因的对应（金字塔 L2 → L3 的桥）

| verdict | 含义 | 下一步 |
|---------|------|--------|
| RACE | kernel 并发竞争（in-place 读改写跨核交错） | 查 decide_inplace_update / buffer 复用（根因4类） |
| REPLAY-MISMATCH | 重放确定但结果与图内不同 | 输入 buffer 的 stride/状态依赖 → 查 layout tag（根因2类）或索引（根因1类） |
| KERNEL OK | 单点正确 | 错误在组合/时序 → 上 L3 逐 buffer 对拍（runtime_buffer_probe 思路） |

## 已知边界

1. **默认 capture 用 llama 1 层真权重**——其他模型必须 `--args-script`（自检会拦住顺序错误）
2. **REPLAY-MISMATCH 的一个良性格式**：kernel 输出到非捕获引用的 buffer（如 extern kernel 后处理）时可能误报——先看 replay 的输出 buffer 是否在 arg_refs 里
3. **exec 产物会触发 async_compile**——需要 NPU 环境完整（与跑模型同环境）
4. num_args（xnumel/r0_numel）从产物源码重解析，产物变更后必须重新 capture

## 与其他工具的关系

- **precision-watchdog**（提案 C）：门禁回答"整体还达标吗"；本工具回答"具体哪个 kernel 坏、坏成什么样"
- **runtime_buffer_probe.py**：L3 层（链上逐 buffer 对拍）的手工版，未工具化——当 test 全 OK 但整体错时用它
- 提案 A（npu-compile-doctor）：把 list→capture→test→buffer 对拍→指纹分类串成全自动流水线，本工具是其 L2 组件
