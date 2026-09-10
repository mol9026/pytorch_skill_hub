# 修改建议 — `{文件名}`

> 由 test-modification-advisor 生成。源文件：`{文件路径}` ｜ 扫描：`scan_combined.py`（= scan_flags + scan_hw_classification）｜ 日期：{YYYY-MM-DD}
> 判据：`../../capability-migrator/references/flag-capability-map.md` §一/§二 + `../../capability-migrator/references/capability-spec.md` §三~§六｜
> `../../hw-classification-annotator/references/hw-signal-map.md` §一 + `../../hw-classification-annotator/references/hw-classification-spec.md` §二~§五｜交互协调：`../references/coordination-guide.md` §一~§六
> 覆盖验证：`references/coverage-preservation.md`。静态计划、collection、runtime verified 与 unresolved 分开记录。

## 0. 概览

- 文件状态：{未迁移 / 部分迁移（哪些类已迁）/ 标注缺失 / 标注有误}
- 总行数：{} ｜ 测试类：{} ｜ 能力 flag 引用：{} ｜ skip/xfail 装饰器：{} ｜ 硬编码设备候选：{} ｜ 已迁移标记：{} ｜ hw 已标注/缺失：{}/{}
- 适用性：{整体适用 / 部分适用（哪些类不适用）/ 不适用（纯逻辑无设备测试）}

## 1. 发现清单（贴 scan_combined.py 输出）

```
{粘贴 python scripts/scan_combined.py <file> 的 markdown 输出}
```

## 2. 差距分析（本地文件 ↔ 目标机制）

### 2.1 能力门控差距（采用 capability 子 skill 的 gate map 建议）

| scan 发现的 gate / flag | 类别 | 能力名 | 状态 | 处理 |
|---|---|---|---|---|
| `PLATFORM_SUPPORTS_FP8` | PLATFORM_SUPPORTS_* | `dtype.fp8` | ✅ 已注册 | 直接用 |
| `requires_cuda_and_triton` | requires_* | `lib.triton` | ✅ 已注册 | 拆两半：triton 部分换装饰器，设备维度走 only_for/except_for |
| `require_dist_backend` | 分布式 backend gate | `distributed.backend` | ✅ 已注册 | 替换为能力门控；保留数量与 world-size gate |
| `TEST_CUDNN` | TEST_* | `lib.cudnn` | ❌ 未注册 | 按 `capability-spec.md` §五注册（改 PyTorch 源码） |
| `TEST_CUDA` | TEST_*（设备可用性） | — | 🚫 非能力 | 走 only_for/except_for（hw 分类一起定） |
| `@skip_if_lt_x_gpu` | 分布式准入 | — | 🚫 非能力 | 原样保留（`flag-capability-map.md` §一明细 4） |
| `TEST_CUDAMALLOCASYNC` | TEST_*（运行环境） | — | 🚫 非能力 | 保留原 skipIf |
| ... | | | | |

### 2.1.1 待定能力

> 仅在 capability 子 skill 无法读取 gate 定义时填写。advisor 只转述建议，不自行命名或定性。

| gate / 符号 | 暂定建议 | 需要核实的条件 | 后续动作 |
|---|---|---|---|
| `{requires_new_feature}` | `Capability.{category.feature}` | `{读取定义后确认实际条件}` | `{确认后复用已有能力 / 按 capability-spec.md §五注册 / 标为非能力}` |

### 2.2 hw_classification 差距（按决策树，以迁移后最终状态判定）

| 测试类 | 当前标注 | 应有分类（依据） | 差距 |
|---|---|---|---|
| `{ClassA}` | {缺失 / GENERIC / ...} | {ACCELERATOR：迁移后含设备方法，跨设备} | {补声明 + 约束修复} |
| `{ClassB}` | ... | {GENERIC：纯逻辑无设备} | {无需能力迁移} |
| ... | | | |

### 2.3 迁移前覆盖基线与迁移后预期矩阵

| 测试 / old→new mapping | Before device/backend/参数/gate/test ID | After 预期 | 变化理由 | 证据状态 |
|---|---|---|---|---|
| `{test_x}` | `{...}` | `{...}` | `{无变化 / 扩大或缩小的依据}` | `{static-plan/collected/runtime-verified/unresolved}` |

## 3. 合并修改动作（按 coordination-guide.md 编辑顺序）

### `class {ClassName}` — hw: {分类} ｜ capability: {能力列表}

**协调说明**：{为什么这么分类（最终状态依据）、为什么这么定 instantiate 参数（同时满足 hw 约束与 capability 设备范围）、是否拆分类}

**执行范围绑定**：{device-type：instantiate + only_for/except_for + device 参数；或 direct multiprocess：device_type/device resolver + backend 范围}

**逐方法同质性**：{设备集合一致 / 必须拆成哪些类}。不得以 ACCELERATOR 吞并 CPU-only/CUDA-only 等异质方法。

**动作清单**：
- {加 hw_classification 声明 + import}
- {删散点 import / 换 @requires_capabilities / 加 device 参数 / 拆分多后端}
- {instantiate 调整：only_for / except_for / 新增调用}
- {让 tensor/module/autocast/stream/event/helper 使用注入设备；设备专属硬编码候选说明}
- {清理旧 flag、冗余 @only*、未使用 import/device、重复测试/helper、失效 skip}

**可直接应用的修改**：
```python
# ================= BEFORE =================
{迁移前代码}
# ================= AFTER ==================
{迁移后代码}
```

（每个待改测试类/方法一节）

## 4. 建议改 PyTorch 源码的项（`capability-spec.md` §五落点 —— 不是测试文件）

> 仅列 capability 子 skill 状态为 ❌ 建议新增能力的项。落点：`torch/testing/_internal/common_device_type.py`（`Capability` 类）+ 各 `*TestBase._capabilities()`。

### 能力 {lib.cudnn}（← `TEST_CUDNN`）

- **步骤1** `Capability` 加常量（`common_device_type.py`）：
  ```python
  class Capability:
      class lib:
          triton = "lib.triton"
          cudnn  = "lib.cudnn"      # ← 新增
  ```
- **步骤2** 各相关后端 `_capabilities()` 加 lambda（组合模式：`super()` 起手 + `update()`）：
  ```python
  # CUDATestBase._capabilities()
  capabilities[Capability.lib].update(
      {Capability.lib.cudnn: lambda: TEST_CUDNN}   # ← 新增
  )
  # CPUTestBase / XPUTestBase 显式 update({Capability.lib.cudnn: lambda: False})
  ```

（按需重复每个未注册能力）

## 5. 不适用项 / 豁免

- `class {TestZzz}(TestCase)`（L{n}）：纯逻辑 / 无设备测试，不迁 capability；hw 标 `GENERIC`。
- 非能力 flag（`TEST_CUDAMALLOCASYNC` / `IS_WINDOWS` / `TEST_WITH_TORCHDYNAMO` / ...）：保留原 skipIf。
- `test/cpython` 目录 hw 侧跳过。
- ⚠️ 待定能力：{gate + 子 skill 的暂定建议 + 待核实条件}；不擅自改写。

## 5.5 覆盖守恒证据

### Collection manifest 差异

- Before 命令/文件：`{...}`
- After 命令/文件：`{...}`
- `compare_collection_manifest.py` 结果：{added/removed/unchanged}
- Added/removed test ID 的逐项解释：{...}

### Runtime 验证

| 环境/device/backend | 命令 | 结果 | 证据状态/限制 |
|---|---|---|---|
| `{cpu/cuda/xpu/...}` | `{...}` | `{pass/skip/fail}` | `{runtime-verified/unresolved}` |

### 残留扫描

- 旧 flag/import：{无 / 清单}
- 冗余 decorator、未用 device/import：{无 / 清单}
- 重复测试/helper、失效 skip、未实例化模板类：{无 / 清单}
- 共享状态恢复：{不涉及 / context manager 或 try/finally 证据}

## 6. 合并检查清单（coordination-guide.md 六）

**先过领域权威清单：**
- [ ] capability subskill：`../../capability-migrator/references/capability-spec.md` §七已通过
- [ ] hw subskill：`../../hw-classification-annotator/references/hw-classification-spec.md` §六已通过

**再过 advisor 交互项：**
- [ ] 分类按迁移后的最终方法集判定；待定或 `UNRESOLVED` 未被 advisor 覆盖
- [ ] `only_for`/`except_for` 已统一定稿，没有重复表达设备范围
- [ ] 同一硬编码设备候选只处理一次；保留的设备专属语义例外已有依据
- [ ] direct multiprocess 未被套用 device-type 的“每方法 `device` 参数”要求
- [ ] 无法同时满足的项已报告“需拆类 / 需确认”，未擅自弱化语义
- [ ] 已记录覆盖基线、差异、runtime 证据和未验证项
