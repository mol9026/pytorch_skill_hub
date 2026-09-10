#!/bin/bash
# NPU 诊断环境信息采集脚本
# 诊断开始时自动执行，无需用户提供

# NPU 型号及固件
npu-smi info -t board -i 0

# CANN 版本
cat /usr/local/Ascend/version.info

# torch 版本
python -c "import torch; print(torch.__version__)"

# torch_npu 版本
pip show torch-npu | grep Version
