#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llava（LlavaForConditionalGeneration 同构）args builder。

TorchBench 原版需 liuhaotian/llava-v1.5-13b 权重（gated，不可达）——
改为 transformers from_config 随机权重同构小模型：CLIP vision + Llama LM + projector。

架构（对四根因的预期）：
  - vision tower: CLIPViT（conv/patch embed + attention）→ 新覆盖面
  - LM: Llama 系（RoPE + SDPA）→ rc1/2/3/4 全部适用面（修复态应覆盖）
  - 多模态拼接（image tokens 嵌入 input_ids 序列）→ reshape/索引链复杂，rc1 观察点
"""

import torch
import torch_npu  # noqa: F401
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration


def build_args(device="npu"):
    torch.manual_seed(42)
    # 小型同构：vision 336×336 输入 / 9 层 ViT；LM 4 层小 Llama
    vision_cfg = CLIPVisionConfig(
        image_size=336, patch_size=14, hidden_size=512,
        intermediate_size=1024, num_hidden_layers=9, num_attention_heads=8,
        num_channels=3,
    )
    text_cfg = LlamaConfig(
        vocab_size=32000, hidden_size=512, intermediate_size=1024,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=8,
        max_position_embeddings=2048,
    )
    cfg = LlavaConfig(vision_config=vision_cfg, text_config=text_cfg)
    model = LlavaForConditionalGeneration(cfg).to(device=device, dtype=torch.float16).eval()

    # 输入：随机像素 + 随机 token 序列（image token 数必须等于 patch 数 576=24×24）
    torch.manual_seed(123)
    bs = 1
    pixel_values = torch.randn(bs, 3, 336, 336, device=device).half()
    img_tok = cfg.image_token_index
    n_patch = (336 // 14) ** 2  # 576
    seq = torch.randint(3, 31000, (bs, 64 + n_patch), device=device)
    seq[0, 10:10 + n_patch] = img_tok  # 连续 576 个 image token
    attn = torch.ones_like(seq)

    with torch.no_grad():
        out = model(pixel_values=pixel_values, input_ids=seq, attention_mask=attn)
        ref = out.logits.detach()

    def forward_fn(m):
        return m(pixel_values=pixel_values, input_ids=seq, attention_mask=attn).logits

    return [pixel_values, seq, attn], ref, model, forward_fn
