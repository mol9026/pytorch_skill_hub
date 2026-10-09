#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BertForMaskedLM（hf_Bert，TorchBench 口径）args builder。

架构（对四根因的预期与测试价值）：
  - **首个双向 attention 架构**：attention mask 只遮 padding（非因果）——mask 形态新覆盖面
  - learned 位置嵌入（非 RoPE）→ rc3 不适用
  - LayerNorm+残差 + GELU → rc4 适用
  - MLM 头（dense+GELU+LayerNorm+decoder 到 vocab）→ 大 vocab 输出投影
"""

import torch
import torch_npu  # noqa: F401
from transformers import BertConfig, BertForMaskedLM


def build_args(device="npu"):
    torch.manual_seed(42)
    # bert-base 同构（12L×768，768 参数量级）；vocab 30522 同原版
    cfg = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                     num_attention_heads=12, intermediate_size=3072)
    model = BertForMaskedLM(cfg).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    input_ids = torch.randint(0, 30500, (1, 512), device=device)
    # 部分位置加 padding 语义（激活双向 mask 路径：attention 只遮 pad）
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 400:] = 0
    token_type_ids = torch.zeros_like(input_ids)
    labels = input_ids.clone()  # MLM 对拍用 logits 即可，labels 不传（避免 loss 路径干扰）

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        ref = out.logits.detach()

    def forward_fn(m):
        return m(input_ids=input_ids, attention_mask=attention_mask,
                 token_type_ids=token_type_ids).logits

    return [input_ids, attention_mask, token_type_ids], ref, model, forward_fn
