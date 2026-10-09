#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""moondream（vikhyatk/moondream1 同构）args builder。

TorchBench 口径：PhiForCausalLM（moondream1 = SigLIP vision + proj + Phi-1.9 LM）。
原版需 gated 权重——手工同构组合：
  SigLIP vision (336px) → linear proj 到 LM 维 → 特征拼进 embeds 序列 → Phi LM

覆盖面：SigLIP（另一支 vision 家族，与 llava 的 CLIP 不同）+ Phi LM
（rotary + partial RoPE，与 Llama 不同的 RoPE 变体）。
"""

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
from transformers import PhiConfig, PhiForCausalLM, SiglipVisionConfig, SiglipVisionModel


class MoondreamLike(nn.Module):
    """SigLIP vision + projector + Phi LM 的 moondream 同构。"""

    def __init__(self, vision_cfg, lm_cfg):
        super().__init__()
        self.vision = SiglipVisionModel(vision_cfg)
        self.vision_proj = nn.Linear(vision_cfg.hidden_size, lm_cfg.hidden_size)
        self.lm = PhiForCausalLM(lm_cfg)
        self.img_tok = 32000  # 占位 image token id（> vocab 边界外用 embedding 容错：用 vocab-1）

    def forward(self, pixel_values, input_ids):
        # vision 特征 (bs, n_patch, v_hidden) → proj 到 LM 维
        vis = self.vision(pixel_values).last_hidden_state  # (bs, P, V)
        vis = self.vision_proj(vis)                        # (bs, P, H)

        # embedding 序列里把 image token 位置替换为 vision 特征
        embeds = self.lm.model.embed_tokens(input_ids)     # (bs, L, H)
        img_mask = input_ids == self.img_tok
        # 逐 batch 拼接（P 个连续 image token 位置）
        bs = input_ids.shape[0]
        for b in range(bs):
            pos = img_mask[b].nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                embeds[b, pos] = vis[b, : len(pos)]

        out = self.lm(inputs_embeds=embeds)
        return out.logits


def build_args(device="npu"):
    torch.manual_seed(42)
    vision_cfg = SiglipVisionConfig(
        image_size=336, patch_size=14, hidden_size=384,
        intermediate_size=1024, num_hidden_layers=4, num_attention_heads=6,
        num_channels=3,
    )
    lm_cfg = PhiConfig(
        vocab_size=32000, hidden_size=512, intermediate_size=1366,
        num_hidden_layers=4, num_attention_heads=8,
        partial_rotary_factor=0.5,  # Phi 特征：部分 RoPE
    )
    model = MoondreamLike(vision_cfg, lm_cfg).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    pixel_values = torch.randn(1, 3, 336, 336, device=device).half()
    n_patch = (336 // 14) ** 2  # 576
    # 注意 vocab 32000：img_tok=32000 超界 → 用 31999 代替（随机权重下语义等价）
    model.img_tok = 31999
    seq = torch.randint(3, 31000, (1, 64 + n_patch), device=device)
    seq[0, 10:10 + n_patch] = model.img_tok

    with torch.no_grad():
        ref = model(pixel_values, seq).detach()

    def forward_fn(m):
        return m(pixel_values, seq)

    return [pixel_values, seq], ref, model, forward_fn
