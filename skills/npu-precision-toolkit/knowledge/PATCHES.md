# 在位修改清单与灾备（PATCHES.md）

> **用途**：① 环境重建/升级后按此恢复全部修复；② 提 PR 时按此核对修改点；
> ③ 红灯测试/回退实验时按此枚举全部触点（2026-09-24 教训：遗漏一个触点 = 假阴性）。
>
> 生成：2026-09-24（watchdog 红灯测试期间审计确认 6 个触点）
> 更新：2026-10-09（rc2 退役：触点 #4/#5 已从 live 恢复上游原版，理由见 #7 行）

## 修改点总表

> **PR 归属标注（2026-10-09）**：#3 已进 gitcode PR [#47232](https://gitcode.com/ascend/pytorch/pulls/47232)；
> #4+#6 的 rc4 部分已收敛为单点修复进 PR [#47261](https://gitcode.com/ascend/pytorch/pulls/47261)；
> #1/#2 待提（见 `docs/PR_PLAN.md`）；rc2 经 2026-10-09 隔离验证取消（见 #7 行）。
> 上游合并前本表仍是服务器环境的权威状态。

| # | live 文件（相对 site-packages） | 修改内容 | 根因 | PR 归属 | backup |
|---|------|------|------|---------|--------|
| 1 | `triton/backends/ascend/utils.py:438` | `-std=c++17` → `-std=c++20` | B1 编译阻断 | 待提（PR-1） | `triton_ascend_utils.py` |
| 2 | `torch_npu/_inductor/select_algorithm.py` | `patch_algorithm_selector.__call__` 加 `**kwargs` | B5 签名不兼容 | 待提（PR-2） | `select_algorithm.py` |
| 3 | `torch_npu/_inductor/triton_experimental/fx_passes.py:479` | `can_merge_dims` 相邻轴守卫（`b-a != 1 → return False`） | rc1 跨轴合并索引错位 | **已提 #47232** | （无 backup，原版无守卫） |
| 4 | ~~`torch_npu/_inductor/triton_experimental/__init__.py`~~ | **已退役（2026-10-09）**：原为 `_install_fa_layout_constraints` monkeypatch（rc2 FA tag + rc4 reduction 跳过）；rc4 部分由 #6 承担，rc2 部分经隔离验证取消。live 文件残留的死函数 `_install_fa_layout_constraints`（无调用者）待环境重建时自然消失 | rc2（已证伪）+ rc4（→#6） | rc4 → **#47261**；rc2 → 取消 | `te_init.py` |
| 5 | ~~`torch_npu/_inductor/__init__.py:383`~~ | **已退役（2026-10-09）**：原为 loader 末尾调用 `_install_fa_layout_constraints()`；live 已恢复上游原版（无任何调用） | #4 的安装点 | 随 #4 退役 | （上游原版即恢复态） |
| 6 | `torch_npu/_inductor/triton_experimental/codegen/triton.py:7718` | `node.decide_inplace_update()` 调用点 → 注释 + `pass`（gating 由 #4 的 monkeypatch 承担） | rc4 第二修复点（双保险） | **已提 #47261**（收敛版：调用侧守卫） | `triton_codegen.py` |
| 7 | （已取消）rc2 生产化：曾计划用上游 `add_layout_constraint` API 为 FA v3 fwd/bwd 注册 `require_contiguous_strides`。**2026-10-09 隔离红绿灯实验后取消**：rc2 完全禁用（rc1/rc4 保留）下前向 strided 直调 cos=1.0、反向 grad cos=1.0、llama16h 全模型 cos=1.004152 与基线一致——aclnn FA v3 实际支持 strided BNSD；历史 K4 的 0.158 系 rc1 未修复期测量污染。方案本身有效，分支留存 worktree `/home/z00599499/pr_work_fa`（commit b15ac29d71）备用 | rc2 | **取消（不提）** | - |

site-packages 绝对路径：`/home/z00599499/envs/miniconda/envs/build/lib/python3.12/site-packages`

## 恢复命令（环境升级/重装后逐条执行）

```bash
SITE=/home/z00599499/envs/miniconda/envs/build/lib/python3.12/site-packages

# #1（一行）
sed -i "s/-std=c++17/-std=c++20/" $SITE/triton/backends/ascend/utils.py

# #2（见 backup diff；原版 __call__ 无 **kwargs）
# 对照 patches_backup/select_algorithm.py 恢复修改

# #3 fx_passes 守卫：can_merge_dims 开头加
#     if b - a != 1:
#         return False
# （恢复"修复前"状态则删除此守卫）

# #4/#5 已退役（2026-10-09）：上游原版即正确状态，无需恢复。
# triton_experimental/__init__.py 若含死函数 _install_fa_layout_constraints（无调用者）
# 可忽略或重装消除；__init__.py loader 保持上游原版。

# #6 codegen/triton.py:7718：node.decide_inplace_update() → 13 行注释 + pass
#    （对照 patches_backup/triton_codegen.py 与 live 的 diff，注释全文在 diff 里）
```

## 验证修复是否在位（一行命令）

```bash
python3 - <<'EOF'
checks = {
    "fx_passes guard": "can_merge_dims" in open("/home/z00599499/envs/miniconda/envs/build/lib/python3.12/site-packages/torch_npu/_inductor/triton_experimental/fx_passes.py").read(),
    "rc2 retired (upstream loader)": "fa_layout" not in open("/home/z00599499/envs/miniconda/envs/build/lib/python3.12/site-packages/torch_npu/_inductor/__init__.py").read(),
    "inplace codegen patch": "decide_inplace_update" in open("/home/z00599499/envs/miniconda/envs/build/lib/python3.12/site-packages/torch_npu/_inductor/triton_experimental/codegen/triton.py").read(),
}
print({k: ("OK" if v else "CHECK") for k, v in checks.items()})
EOF
# 或直接跑回归门禁（最可靠）：
#   python3 scripts/precision_watchdog.py --gate g12   # 修复态应全 PASS
```

## 红灯测试操作记录（2026-09-24，供复现）

全撤 #3/#5/#6 + `rm -rf /tmp/torchinductor_root`（缓存不感知 monkeypatch，必须清）→
G4 cos=None(NaN) FAIL exit=1（历史故障完整复现）。恢复后 G4=1.000014 PASS。
仅撤 #5（保留 #6）时故障不复现（cos=1.000016）——**#4/#6 是双保险，回退实验必须两个都撤**。
