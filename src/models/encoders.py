"""Bare vision encoders: the T0-c reference ceiling and the T0 baseline (§5, §13).

T0-c and T0 come from **different tensors** and must never be conflated:

* ``T0-c`` is the contrastive embedding — ViT *last* layer CLS pushed through
  ``visual_projection`` and L2-normalised. It is the only tap where zero-shot
  text matching means anything, and it is *not* what LLaVA consumes.
* ``T0`` is the penultimate-layer patch grid, CLS dropped — what the projector
  actually sees.

Reporting a T0-c number as the "before projection" bar is how you accidentally
report a much larger projector drop than exists.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor

from .base import TapExtractor, Taps


class EncoderTapExtractor(TapExtractor):
    """CLIP-ViT-L/14-336 or SigLIP2-so400m-p14-384, contrastive head included."""

    has_llm = False

    def __init__(
        self,
        hf_id: str = "openai/clip-vit-large-patch14-336",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        select_layer: int = -2,
        has_cls: bool = True,
    ):
        from transformers import AutoModel, AutoProcessor

        self.name = hf_id.split("/")[-1]
        self.hf_id = hf_id
        self.device = device
        self.dtype = dtype
        self.select_layer = select_layer
        # SigLIP has no CLS token: its pooled vector comes from an attention-pooling
        # head over the patch grid, so every hidden-state position is a patch.
        self.has_cls = has_cls

        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id, dtype=dtype).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        vcfg = self.model.config.vision_config
        self.d_vis = vcfg.hidden_size
        self.d_contrastive = getattr(self.model.config, "projection_dim", self.d_vis)
        self.m = None

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> Dict[str, Optional[Tensor]]:
        """``t0c`` ``[B, d_c]``, ``t0`` ``[B, m, d_vis]``, ``t0_cls`` ``[B, d_vis]`` or None."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(self.dtype)

        vout = self.model.vision_model(pixel_values, output_hidden_states=True)
        hs = vout.hidden_states[self.select_layer]
        if self.has_cls:
            t0, t0_cls = hs[:, 1:], hs[:, 0]
        else:
            # SigLIP has no CLS position at all. Returning the patch mean here
            # would write a "T0.cls" file byte-identical to T0.avg and invite
            # someone to compare them as if they were different taps.
            t0, t0_cls = hs, None
        if self.m is None:
            self.m = t0.shape[1]

        # get_image_features runs the model's own contrastive head (last layer +
        # pooling + visual_projection), which is the definition of T0-c.
        t0c = self.model.get_image_features(pixel_values=pixel_values)
        t0c = t0c / t0c.norm(dim=-1, keepdim=True)
        return {"t0c": t0c, "t0": t0, "t0_cls": t0_cls}

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 256) -> Tensor:
        """L2-normalised text embeddings, for zero-shot classification (§6.1)."""
        outs = []
        for i in range(0, len(texts), batch_size):
            enc = self.processor(
                text=texts[i : i + batch_size],
                return_tensors="pt",
                padding="max_length" if "siglip" in self.hf_id.lower() else True,
                truncation=True,
                max_length=64,
            ).to(self.device)
            enc.pop("pixel_values", None)
            f = self.model.get_text_features(**enc)
            outs.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(outs)

    def forward_taps(
        self, images: List[Image.Image], prompt: str, layers: Sequence[int] = ()
    ) -> Taps:
        raise NotImplementedError(
            "encoders have no LLM stack; use encode_images() and extract T0-c/T0/T0.cls"
        )

    def manifest(self) -> Dict[str, Any]:
        d = super().manifest()
        d.update(
            hf_id=self.hf_id,
            select_layer=self.select_layer,
            has_cls=self.has_cls,
            d_contrastive=self.d_contrastive,
            dtype=str(self.dtype),
        )
        return d
