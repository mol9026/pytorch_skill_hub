---
name: npu-fleet-ops
description: >-
  NPU 服务器机群运维技能：多机 SSH 接入（config 别名 + 公钥分发）、
  资源审计五件套（CPU/内存/NPU 卡/磁盘/负载）、磁盘病灶定位
  （/root 缓存 vs docker vs /home 物主）、备份与恢复双端校验、
  重装迁移清单（/mnt/weight 权重 + /mnt/share 数据分工）。
  当用户要接入新服务器、盘点机器资源、排查磁盘满、做备份/重装迁移时使用。
  Use when onboarding new NPU servers, auditing fleet resources,
  diagnosing full disks, or planning backup/migration before OS reinstall.
---

# npu-fleet-ops：NPU 服务器机群运维

## 概述

多台 NPU 服务器（共享、多团队）的日常运维流程。源自 4 台 910B 机器的实战：
两次全量备份（零流程修改）、一次系统盘病灶定位（四台交叉验证）、一次重装前迁移。

**核心原则**：
1. **接入走 config 别名**——禁止裸连（`-i` + IP + flags 满天飞），`~/.ssh/config` 一处管理
2. **共享机先确认归属再动手**——/home 下几十个工号目录，删错就是事故
3. **备份必做双端校验**——tar + md5 服务器/本地各一次，不一致=没备份

## 1. 新服务器接入（每台 2 分钟）

### 1.1 SSH config 别名

追加到本机 `~/.ssh/config`：

```sshconfig
Host npuXXX               # 别名：npu + IP 末段
    HostName 192.168.9.XXX
    User root             # 或个人账号
    IdentityFile ~/.ssh/id_ed25519_npu
    StrictHostKeyChecking no
    ServerAliveInterval 60
```

### 1.2 公钥分发（有密码时）

本机无 sshpass 时的替代——paramiko 一次性脚本：

```python
import paramiko, os
pub = open(os.path.expanduser("~/.ssh/id_ed25519_npu.pub")).read().strip()
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("<IP>", username="root", password="<密码>", timeout=15)
cmd = ('mkdir -p ~/.ssh && chmod 700 ~/.ssh && '
       f'grep -qF "{pub}" ~/.ssh/authorized_keys 2>/dev/null || '
       f'echo "{pub}" >> ~/.ssh/authorized_keys; '
       'chmod 600 ~/.ssh/authorized_keys; echo KEY_INSTALLED')
_, out, _ = c.exec_command(cmd); print(out.read().decode())
c.close()
```

验证：`ssh npuXXX hostname`（免密即成功）。

## 2. 资源审计五件套（每台一条命令）

```bash
ssh npuXXX '
hostname
lscpu | grep -E "^CPU\(s\)|Model name" | head -2
free -g | grep Mem
echo "-- NPU --"
ls /dev/davinci* 2>/dev/null | wc -l | xargs echo "davinci:"
npu-smi info 2>/dev/null | grep -E "910B|910 " | head -8   # 型号+健康+占用
df -h / /home 2>/dev/null | grep -v tmpfs                   # 注意：/ 与 /home 独立分区！
uptime
ps aux --sort=-%cpu | head -4 | tail -3                     # 谁在用
ls /usr/local/Ascend 2>/dev/null                            # CANN 版本线索
'
```

**读数要点**：
- NPU 型号决定 HBM 容量（910B2=64G / 910B4=32G）——B4 跑 7B 全模型推理紧张
- `/` 与 `/home` 是 **LVM 独立分区**，互不影响；系统盘 93-100% 的病灶**从来不在 /home**

## 3. 磁盘病灶定位（系统盘满时）

按命中率排序的排查顺序：

```bash
du -xh -d1 / 2>/dev/null | sort -rh | head -6        # -x 只看本文件系统
du -h -d1 /root 2>/dev/null | sort -rh | head -6     # 命中率最高：开发缓存
du -h -d1 /var/lib/docker 2>/dev/null | sort -rh | head -3
```

**实测病灶分布**（4 台 910B 机器，openEuler）：

| 病灶 | 典型大小 | 处置 | 风险 |
|------|---------|------|------|
| `/root/.{vscode,trae,cursor}-server` | 5-9G/个 | 删（下次用时重装） | 低 |
| `/root/.cache` + `.triton` + `.ccache` | 5-15G | 删（triton 首次编译变慢） | 低 |
| `/var/lib/docker/overlay2` | 40G+ | `docker system prune`；**先 `docker ps -a` 核容器归属**；部分集群 docker 实际在 /home/docker | 中 |
| `/home/<工号>` 巨型目录 | 数百 G | **找物主**，不属于你的不要动 | 高 |

## 4. 备份与恢复（双端校验）

```bash
# 服务器侧打包（排除清单：模型权重、conda、clone 的大仓、旧备份自身）
cd /home && tar --exclude='<user>/llama-2-7b' --exclude='<user>/envs' \
  --exclude='<user>/pytorch' --exclude='<user>/benchmark' \
  -czf /tmp/<user>_backup_$(date +%Y%m%d).tar.gz <user>/

# 本地拉取 + 双端 md5（不一致 = 没备份成功）
scp npuXXX:/tmp/<user>_backup_*.tar.gz "<本地备份目录>/"
md5sum 服务器文件  vs  certutil -hashfile 本地文件 MD5

# 服务器 /tmp 清理（/tmp 会被清，别放长期数据）
rm -f /tmp/<user>_backup_*.tar.gz
```

**保留策略**：最近 1-2 个日期版本，更早的确认后删。

## 5. 重装迁移（/mnt 分工）

重装/换机前，数据按性质分流到 NFS 共享盘（跨机器可见，注意**是共享的——别人的误删也会影响你**）：

| 数据 | 去向 | 说明 |
|------|------|------|
| 模型权重 | `/mnt/weight/<工号>/` | 大文件只读资产，恢复时软链不复制 |
| 代码/文档/证据/环境清单 | `/mnt/share/<工号>/{docs,scripts,evidence,env_manifest}/` | `pip freeze` 必存 |
| conda env 整包（可选） | `/mnt/share/<工号>/build_env.tar.gz` | `conda pack`；**前提：重装后 CANN 版本一致** |
| 不备份 | /tmp 产物、可重编译的 torch_compile_debug、可重新 clone 的大仓 | — |

**重装前 3 天完成备份 + 恢复演练**（解包抽查关键文件），不要拖到最后一天。

**恢复验收门禁**：环境重建后跑精度门禁（如 `precision_watchdog --gate g12`）全 PASS，才可与重装前等价。

## 6. 踩坑沉淀

1. **/tmp 是临时的**：会被系统清理，PR 工作仓库放 /tmp 曾丢失（好在分支推上了远端）。仓库类放 /home 或 /mnt
2. **共享机的 /home 有几十个工号目录**：动手前 `ls /home` + 归属确认；别人的巨型目录（实测单目录 818G）只能找物主
3. **docker 数据可能在 /home/docker 也可能在 /var/lib/docker**：`docker info | grep "Docker Root Dir"` 先看真实位置再清理
4. **系统盘满与 /home 无关**：LVM 独立分区；病灶在 /root 缓存、docker、/var/log
5. **NFS 共享盘是公共的**：备份到 /mnt/share 防重装，但防不了同事误删——关键证据在恢复后于 /home 落一份工作副本
6. **别人的长任务**：盘点时看到 119% CPU 的 pytest 跑了几天——共享机上动任何东西前先 `ps aux` 看归属

## 7. 验证（skill 可用判据）

- 新机器接入后 `ssh <别名> hostname` 免密通过
- 审计命令在任一机器跑通且五件套输出齐全
- 备份文件双端 md5 一致
