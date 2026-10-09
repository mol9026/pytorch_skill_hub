---
name: server-workspace-hygiene
description: >-
  NPU 排查服务器的工作目录整理与本地备份技能：目录巡检（scripts/evidence/debug
  归位）、debug 产物按时间窗归档、tar 备份带日期、本地拉取与校验、交付文档双端同步。
  当用户说"整理服务器"/"备份到本地"/"服务器乱不乱"/阶段收尾需要固化工作成果时使用。
  Use when the NPU debugging server needs workspace cleanup, archiving of
  stale debug runs, dated backup tarballs pulled to the local PC, or
  delivery-doc synchronization between server and PC.
---

# server-workspace-hygiene：服务器工作区整理与本地备份

## 概述

NPU 排查工作会产生三类膨胀物：编译 debug 产物（每次 run 数 MB，积累最快）、
实验脚本（散在 /tmp 与家目录）、证据与文档（双端各一份易漂移）。本 skill 是
两轮实战（2026-09-28/09-29）收敛出的标准流程。

## 前提

- SSH 到 NPU 服务器（按你的环境替换 `<user>@<server>`；示例业务目录 `/home/<user>/`）
- 本地有备份目标目录（示例：`Desktop/<项目备份目录>/04_服务器备份/`）

## 标准目录结构（服务器侧目标态）

```
/home/<user>/
├── scripts/                  # 工具脚本（可执行，团队共享）
│   └── puncture_args/        # 穿刺 args 模板集中管理
├── evidence/<项目名>/         # 实验证据（JSON + log，只增不删）
├── watchdog_results/<项目名>/ # 门禁结果
├── docs/                     # 交付文档副本（与本地 00_交付文档 同步）
├── patches_backup/           # site-packages 在位修改的灾备
├── torch_compile_debug/      # 近期编译产物（保留 ~2 天）
├── torch_compile_debug_archive/  # 归档区（压缩后，供回溯）
└── benchmark/                # 基准数据
```

## 流程（四步）

### 1. 巡检：先看再动

```bash
# 家目录全貌 + 大头排行
ls /home/<user>/
du -sh /home/<user>/* | sort -rh | head -10
du -sh /home/<user>/torch_compile_debug | sort -rh
ls /home/<user>/scripts/ | head -30   # 找散落的 *_args.py、临时脚本
```

判断规则：
- debug run 目录按 mtime 超 2 天 → 归档候选
- 散落的 `*_args.py` / 一次性脚本 → 移入 `scripts/puncture_args/` 或删除（verdict 类留 evidence）
- 未知文件先查内容再动（可能是别人/别的会话的在工作产物——**服务器可能是共享的，
  动手前确认归属**）

### 2. 归档：debug 产物瘦身

```bash
cd /home/<user>
# 保留最近 N 个 run（近期排查可能还要回看），其余打包进归档区
ls -t torch_compile_debug/ | tail -n +9 | while read d; do
  tar -czf "torch_compile_debug_archive/${d}.tar.gz" -C torch_compile_debug "$d" \
    && rm -rf "torch_compile_debug/$d"
done
du -sh torch_compile_debug torch_compile_debug_archive
```

实测收益：43M → 25M（96 run 归档，8 run 保留）。

### 3. 备份：tar + 日期 + 本地拉取

```bash
# 服务器侧打包（排除超大无关物：模型权重、conda、旧备份自身）
cd /home && tar -czf /tmp/<user>_backup_$(date +%Y%m%d).tar.gz \
  --exclude='<user>/llama-2-7b' \
  --exclude='<user>/envs' \
  --exclude='<user>/pytorch' \
  <user>/

# 本地拉取
scp <user>@<server>:/tmp/<user>_backup_$(date +%Y%m%d).tar.gz \
  "<本地备份目录>/"

# 校验（两端的 md5 必须一致）
md5sum /tmp/<user>_backup_*.tar.gz   # 服务器
certutil -hashfile "<本地备份目录>\<同名>" MD5   # Windows 本地
```

替换策略：新包完整覆盖旧包（同一天重做则同名覆盖）。保留最近 1-2 个日期版本即可，
更早的删（先确认用户）。实测单包 ~109M（含全部穿刺证据）。

### 4. 同步：交付文档双端一致

```bash
# 改动侧 → 另一侧；关键文档（PATCHES.md / 台账 / PR_PLAN）必须双端同版
scp "<本地>/00_交付文档/PATCHES.md" <user>@<server>:/home/<user>/patches_backup/
scp "<本地>/00_交付文档/PATCHES.md" <user>@<server>:/home/<user>/docs/
```

校验：`diff` 或 head 比对两端的"更新日期"行。

## 踩坑沉淀

1. **evidence 只增不删**——verdict/裁决 JSON 是归因修正的证据链（rc2/rc3 的翻案
   全靠它们），宁可占空间。
2. **/tmp 会被服务器清理**——PR 工作仓库放 /tmp 曾被清掉（分支在 fork 上才没丢）。
   仓库类放家目录持久区；/tmp 只放当天中间产物。
3. **共享服务器先确认归属**——别人的构建区（如 envs/pkgs/ 下的 worktree）不能动；
   建 git worktree 用 `git worktree add` 隔离，不动主工作区。
4. **备份排除清单**——模型权重（GB 级、可重下）、conda envs（可重建）、旧备份自身。
5. **tar 前先 du 估算**——排除清单漏一项，备份从 109M 涨到 10G+。

## 验证（skill 装好的判据）

```bash
# 服务器侧：结构完整
ls /home/<user>/{scripts,evidence,docs,patches_backup,watchdog_results} -d
# 归档生效
du -sh torch_compile_debug  # 应显著小于归档前
# 本地侧：备份可解压
tar -tzf "<备份目录>"/<user>_backup_*.tar.gz | head -5
```
