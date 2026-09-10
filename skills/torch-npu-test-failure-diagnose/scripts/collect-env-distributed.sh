#!/bin/bash
# 分布式诊断环境信息采集脚本
# 在分布式用例诊断时额外执行

# NPU 设备信息
npu-smi info -t board -i 0 2>/dev/null || echo "npu-smi not available"
npu-smi info -t health -i 0 2>/dev/null

# HCCL 库状态
ldconfig -p | grep -i hccl 2>/dev/null
ls /usr/local/Ascend/*/lib64/libhccl.so 2>/dev/null

# 分布式环境变量
echo "MASTER_ADDR=${MASTER_ADDR:-<not set>}"
echo "MASTER_PORT=${MASTER_PORT:-<not set>}"
echo "WORLD_SIZE=${WORLD_SIZE:-<not set>}"
echo "RANK=${RANK:-<not set>}"

# torch_npu 分布式支持
python -c "from torch_npu.contrib import transfer_to_npu; print('transfer_to_npu OK')"
python -c "import torch.distributed as dist; print('HCCL available:', dist.is_hccl_available())"
