"""LLaVA-1.5 tap extractor, built on the HF ``LlavaForConditionalGeneration`` port.

Why the HF port and not ``haotian-liu/LLaVA`` (§5.1): the HF processor expands the
``<image>`` placeholder inside ``input_ids``, so the image-token span is simply
``(input_ids == image_token_id).nonzero()`` — no patching of
``prepare_inputs_labels_for_multimodal``, and no reconstruction that silently
breaks under tiling.

Tap mechanics, all from one pass:

* ``T0``  — vision tower ``hidden_states[vision_feature_layer]``, CLS dropped.
* ``T1``  — ``multi_modal_projector(T0)``. Also readable live as the LLM's
  *layer-0* hidden state at the image positions, which is what §11 check 4 uses.
* ``T2``  — last LLM layer sliced to the image span.
* ``T3``  — last LLM layer, full sequence.
* ``mid`` — ``hidden_states[k]`` at the image span, for the §8.3B layer sweep.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor

from .base import TapExtractor, Taps


def _find_submodule(root: torch.nn.Module, names: Sequence[str]) -> torch.nn.Module:
    """Resolve the first attribute path that exists.

    ``transformers`` moved ``vision_tower`` / ``multi_modal_projector`` under an
    inner ``LlavaModel`` in 4.52. Trying both keeps this working either side of
    that refactor instead of pinning us to one point release.
    """
    for name in names:
        obj: Any = root
        for part in name.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"none of {list(names)} found on {type(root).__name__}")


class LLaVATapExtractor(TapExtractor):
    def __init__(
        self,
        hf_id: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        expect_select_layer: int = -2,
        expect_select_feature: str = "patch",
    ):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.name = hf_id.split("/")[-1]
        self.hf_id = hf_id
        self.device = device
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = LlavaForConditionalGeneration.from_pretrained(
            hf_id, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        cfg = self.model.config
        self.vision_feature_layer = cfg.vision_feature_layer
        self.select_strategy = cfg.vision_feature_select_strategy
        self.image_token_id = getattr(cfg, "image_token_index", None)
        if self.image_token_id is None:
            self.image_token_id = cfg.image_token_id

        # §4: assert rather than assume. Reading the last ViT block instead of the
        # penultimate one would compare against features LLaVA never sees, and the
        # T0->T1 drop would be an artefact.
        assert self.vision_feature_layer == expect_select_layer, (
            f"{hf_id}: vision_feature_layer={self.vision_feature_layer}, "
            f"expected {expect_select_layer}"
        )
        strategy_is_patch = self.select_strategy in ("default", "patch")
        assert (expect_select_feature == "patch") == strategy_is_patch, (
            f"{hf_id}: vision_feature_select_strategy={self.select_strategy!r} "
            f"does not match expected feature {expect_select_feature!r}"
        )

        self.vision_tower = _find_submodule(self.model, ["model.vision_tower", "vision_tower"])
        self.projector = _find_submodule(
            self.model, ["model.multi_modal_projector", "multi_modal_projector"]
        )

        self.d_vis = cfg.vision_config.hidden_size
        self.d_llm = cfg.text_config.hidden_size
        self.n_layers = cfg.text_config.num_hidden_layers
        self.m = None  # filled in on the first forward and asserted stable thereafter

    # ------------------------------------------------------------------ helpers

    def _spans_from_input_ids(self, input_ids: Tensor) -> List[Tuple[int, int]]:
        """Per-example image-token span, asserted contiguous.

        LLaVA-1.5 splices one contiguous run of image tokens. A non-contiguous run
        would mean tiling or a processor change, and every "slice the image span"
        assumption downstream would be wrong — so fail loudly rather than pool
        garbage.
        """
        spans: List[Tuple[int, int]] = []
        for row in input_ids:
            idx = (row == self.image_token_id).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                raise RuntimeError(
                    f"no image token (id={self.image_token_id}) in input_ids; "
                    "the processor did not expand the <image> placeholder"
                )
            start, end = int(idx[0]), int(idx[-1]) + 1
            if idx.numel() != end - start:
                raise RuntimeError(
                    f"image tokens are not contiguous ({idx.numel()} tokens spanning "
                    f"{end - start} positions) — any-res tiling is not supported here"
                )
            spans.append((start, end))
        return spans

    def _vision_features(self, pixel_values: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(patch_tokens, cls_token)`` from the selected ViT layer."""
        out = self.vision_tower(pixel_values, output_hidden_states=True)
        hs = out.hidden_states[self.vision_feature_layer]  # [B, 1+m, d_vis]
        return hs[:, 1:], hs[:, 0]

    def _gather_span(self, x: Tensor, spans: List[Tuple[int, int]]) -> Tensor:
        """Slice ``[B, T, d]`` down to the per-example image span -> ``[B, m, d]``."""
        widths = {e - s for s, e in spans}
        assert len(widths) == 1, f"ragged image spans in one batch: {sorted(widths)}"
        return torch.stack([x[i, s:e] for i, (s, e) in enumerate(spans)])

    # -------------------------------------------------------------------- taps

    @torch.no_grad()
    def forward_taps(
        self,
        images: List[Image.Image],
        prompt: str,
        layers: Sequence[int] = (),
        want_t1_live: bool = False,
    ) -> Taps:
        # Right padding + explicit per-example lengths. The prompt is fixed and m is
        # fixed for LLaVA-1.5, so in practice nothing pads; the machinery is here so
        # that a ragged batch degrades into correct pooling rather than silent drift.
        self.tokenizer.padding_side = "right"
        batch = self.processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        pixel_values = batch["pixel_values"].to(self.dtype)
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        seq_len = attn.sum(1).tolist()

        spans = self._spans_from_input_ids(input_ids)
        t0, t0_cls = self._vision_features(pixel_values)
        if self.m is None:
            self.m = t0.shape[1]
        assert t0.shape[1] == self.m, f"image-token count changed: {t0.shape[1]} != {self.m}"

        t1 = self.projector(t0)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states  # tuple len n_layers+1; [0] is the embedding output
        t3 = hidden[-1]
        t2 = self._gather_span(t3, spans)

        mid: Dict[int, Tensor] = {}
        for k in layers:
            if not 0 <= k < len(hidden):
                raise ValueError(f"layer {k} out of range for {len(hidden) - 1} LLM layers")
            mid[int(k)] = self._gather_span(hidden[k], spans)

        extra: Dict[str, Any] = {}
        if want_t1_live:
            # hidden[0] is the embedding-layer output, so at the image positions it
            # *is* the projected image embedding. Comparing it against the standalone
            # projector(T0) is §11 check 4 with no extra forward pass.
            extra["t1_live"] = self._gather_span(hidden[0], spans)

        taps = Taps(
            t0=t0,
            t1=t1,
            t2=t2,
            t3=t3,
            img_span=spans,
            seq_len=seq_len,
            t0_cls=t0_cls,
            mid=mid,
            extra=extra,
        )
        taps.validate()
        return taps

    @torch.no_grad()
    def forward_text_taps(self, prompts: List[str]) -> Tuple[Tensor, List[int]]:
        self.tokenizer.padding_side = "right"
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1], enc["attention_mask"].sum(1).tolist()

    @torch.no_grad()
    def generate(
        self, images: List[Image.Image], prompt: str, max_new_tokens: int = 32
    ) -> List[str]:
        # Generation needs left padding: with right padding the model would decode
        # from a pad token.
        self.tokenizer.padding_side = "left"
        batch = self.processor(
            images=images, text=[prompt] * len(images), return_tensors="pt", padding=True
        ).to(self.device)
        batch["pixel_values"] = batch["pixel_values"].to(self.dtype)
        ids = self.model.generate(
            **batch,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new = ids[:, batch["input_ids"].shape[1]:]
        return [t.strip() for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]

    def manifest(self) -> Dict[str, Any]:
        d = super().manifest()
        d.update(
            hf_id=self.hf_id,
            vision_feature_layer=self.vision_feature_layer,
            vision_feature_select_strategy=self.select_strategy,
            image_token_id=self.image_token_id,
            span_derivation="input_ids == image_token_id (HF port)",
            dtype=str(self.dtype),
        )
        return d
