#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T5（T5ForConditionalGeneration，TorchBench 口径）args builder。

架构（对四根因的预期）：
  - 相对位置偏置（relative_position_bias，attention bias 表）→ 无 RoPE，rc3 不适用
  - 手写 attention（scores + bias + softmax，非 SDPA）→ rc2 路径待观察
  - RMSNorm+残差 → rc4 适用
  - encoder-decoder cross-attention → 新覆盖面
"""

import torch
import torch_npu  # noqa: F401
from transformers import T5Config, T5ForConditionalGeneration


def build_args(device="npu"):
    torch.manual_seed(42)
    # 小型 T5（原版 t5-small 6+6L×512 同构；用 2+2 层加速编译验证，层深不影响根因覆盖判断）
    cfg = T5Config(num_layers=2, num_decoder_layers=2, d_model=512,
                   d_ff=1024, num_heads=8, vocab_size=32128)
    model = T5ForConditionalGeneration(cfg).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    input_ids = torch.randint(0, 32100, (1, 128), device=device)
    decoder_input_ids = torch.randint(0, 32100, (1, 64), device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        ref = out.logits.detach()

    def forward_fn(m):
        return m(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    return [input_ids, decoder_input_ids], ref, model, forward_fn
