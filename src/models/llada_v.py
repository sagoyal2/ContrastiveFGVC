"""LLaDA-V (V-DLM) tap extractor.

Three things make this model different from the LLaVA path, and all three are
consequences of what its published ``config.json`` actually says:

1. **No HF port.** ``GSAI-ML/LLaDA-V`` ships ``config.json`` +
   ``configuration_llada.py`` but no ``modeling_llada.py``; the modelling code
   lives in ``github.com/ML-GSAI/LLaDA-V`` (a LLaVA-NeXT fork pinned near
   ``transformers==4.39``). It therefore needs its own environment — see
   ``scripts/setup_llada_env.sh``. Everything in this file is written against
   that fork's API.

2. **``m`` is variable.** The config sets ``image_aspect_ratio=anyres_max_4``,
   ``mm_patch_merge_type=spatial_unpad`` and ``mm_newline_position=grid``. So an
   image becomes 1–5 SigLIP2 tiles of 729 tokens, then ``spatial_unpad`` drops
   padding tiles and splices newline tokens in. The count reaching the LLM is
   per-image and is *not* equal to the raw patch count. Never hardcode it (§13).
   ``--llada-aspect pad`` forces the single-tile path when you want a fixed
   ``m=729`` that is structurally comparable to LLaVA's 576.

3. **Attention is bidirectional.** "The hidden state" is a function of the
   denoising timestep and mask pattern, so the tap has to be chosen and
   documented (§9). Hence ``--llada-mode {clean,masked,sweep}``.

   Because attention is not causal, the *last* image token is not privileged the
   way it is in LLaVA: expect ``T2.avg ≈ T2.final`` here. If you see a LLaVA-like
   gap, the span slicing is wrong (§11 check 6).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor

from .base import TapExtractor, Taps

IMAGE_TOKEN_INDEX = -200  # LLaVA-NeXT sentinel for the un-expanded <image> placeholder
DEFAULT_MASK_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)


class LLaDAVTapExtractor(TapExtractor):
    def __init__(
        self,
        hf_id: str = "GSAI-ML/LLaDA-V",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        mode: str = "clean",
        mask_k: int = 8,
        mask_ratios: Sequence[float] = DEFAULT_MASK_RATIOS,
        aspect_ratio: Optional[str] = None,
        conv_template: str = "llada",
        attn_implementation: str = "sdpa",
    ):
        try:
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.model.builder import load_pretrained_model
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "LLaDA-V needs the ML-GSAI/LLaDA-V package (a LLaVA-NeXT fork) on the "
                "path; the HF repo ships no modeling_*.py. Run "
                "scripts/setup_llada_env.sh and use .venv-llada/bin/python."
            ) from exc

        from llava.model.language_model.modeling_llada import LLaDAModelLM

        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._base_lm_class = LLaDAModelLM

        assert mode in ("clean", "masked", "sweep"), f"unknown --llada-mode {mode!r}"
        self.name = "llada-v-8b"
        self.hf_id = hf_id
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.mask_k = mask_k
        self.mask_ratios = tuple(mask_ratios)
        self.conv_template = conv_template

        # Two API details of this fork's builder, both verified against
        # train/llava/model/builder.py rather than assumed:
        #   * torch_dtype is a *string* ("bfloat16"), not a torch.dtype. Passing a
        #     dtype object falls through their if/elif chain into a literal
        #     pdb.set_trace(), which hangs the process with no error.
        #   * attn_implementation defaults to "flash_attention_2". flash-attn is
        #     deliberately not installed in .venv-llada (training-only, needs
        #     compilation against an old torch), so ask for sdpa.
        _DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float16: "float16"}
        dtype_name = _DTYPE_NAMES.get(dtype)
        assert dtype_name is not None, f"LLaDA-V builder takes bf16/fp16 only, got {dtype}"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            hf_id,
            None,
            "llava_llada",
            torch_dtype=dtype_name,
            device_map=device,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        cfg = self.model.config
        # §4 told us to verify these rather than assume them; they hold for the
        # released checkpoint, and the assert keeps a future revision honest.
        assert cfg.mm_vision_select_layer == -2, (
            f"LLaDA-V mm_vision_select_layer={cfg.mm_vision_select_layer}, expected -2"
        )
        assert cfg.mm_vision_select_feature == "patch", (
            f"LLaDA-V mm_vision_select_feature={cfg.mm_vision_select_feature!r}, expected 'patch'"
        )
        self.native_aspect_ratio = cfg.image_aspect_ratio
        self.aspect_ratio = aspect_ratio or cfg.image_aspect_ratio
        self.patch_merge_type = getattr(cfg, "mm_patch_merge_type", "flat")

        self.d_vis = cfg.mm_hidden_size
        self.d_llm = cfg.hidden_size
        self.n_layers = cfg.num_hidden_layers
        self.m = None  # variable under any-res; recorded per example instead

        self.vision_tower = self.model.get_vision_tower()
        self.projector = self.model.get_model().mm_projector
        self.mask_token_id = self._resolve_mask_token_id()

    # ------------------------------------------------------------------ helpers

    def _resolve_mask_token_id(self) -> Optional[int]:
        """LLaDA's ``[MASK]`` id, needed by the ``masked``/``sweep`` modes."""
        for attr in ("mask_token_id", "mask_id"):
            v = getattr(self.model.config, attr, None)
            if isinstance(v, int):
                return v
        for tok in ("<|mdm_mask|>", "[MASK]", "<mask>"):
            ids = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(ids, int) and ids >= 0 and ids != self.tokenizer.unk_token_id:
                return ids
        return None

    def _prepare(self, images: List[Image.Image], prompt: str):
        """Tokenise + tile, then hand to the model's own multimodal splicer.

        This is spec §5.1 option 2 done safely: we let the fork build
        ``inputs_embeds``, then *derive* the span from the placeholder position and
        the length change, and assert the arithmetic. Nothing is guessed.
        """
        cfg = self.model.config
        old_aspect = cfg.image_aspect_ratio
        cfg.image_aspect_ratio = self.aspect_ratio
        try:
            pixel_values = self._process_images(images, self.image_processor, cfg)
        finally:
            cfg.image_aspect_ratio = old_aspect

        if isinstance(pixel_values, list):
            pixel_values = [p.to(self.device, dtype=self.dtype) for p in pixel_values]
        else:
            pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        ids = [
            self._tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for _ in images
        ]
        input_ids = torch.stack(ids).to(self.device)
        image_sizes = [im.size for im in images]
        return input_ids, pixel_values, image_sizes

    def _spans_from_length(self, input_ids: Tensor, embeds_len: int) -> List[Tuple[int, int]]:
        """Reconstruct the image span from the placeholder index and length delta.

        Guarded by the §5.1 assert: the sequence must have grown by exactly
        ``m_i - 1`` tokens, otherwise our idea of what was spliced is wrong.
        """
        spans = []
        for row in input_ids:
            pos = (row == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            assert pos.numel() == 1, f"expected exactly one <image> placeholder, got {pos.numel()}"
            p = int(pos[0])
            m_i = embeds_len - (row.shape[0] - 1)
            assert m_i > 0, f"spliced sequence shrank: embeds={embeds_len} ids={row.shape[0]}"
            spans.append((p, p + m_i))
        return spans

    # -------------------------------------------------------------------- taps

    @torch.no_grad()
    def forward_taps(
        self,
        images: List[Image.Image],
        prompt: str,
        layers: Sequence[int] = (),
    ) -> Taps:
        if self.mode == "sweep":
            raise ValueError("sweep mode returns many tap sets; call forward_taps_sweep()")
        return self._forward_once(images, prompt, layers, mask_ratio=None)

    @torch.no_grad()
    def forward_taps_sweep(
        self, images: List[Image.Image], prompt: str, layers: Sequence[int] = ()
    ) -> Dict[float, Taps]:
        """§9 ``--llada-mode sweep``: one tap set per answer-span mask ratio."""
        return {r: self._forward_once(images, prompt, layers, mask_ratio=r) for r in self.mask_ratios}

    def _answer_span_ids(self, batch: int, mask_ratio: Optional[float]) -> Optional[Tensor]:
        """Build the appended answer span for the masked / sweep modes."""
        if self.mode == "clean" and mask_ratio is None:
            return None
        if self.mask_token_id is None:
            raise RuntimeError(
                "could not resolve LLaDA's [MASK] token id; --llada-mode masked/sweep "
                "cannot run. Inspect tokenizer_config.json and pass it explicitly."
            )
        k = self.mask_k
        ratio = 1.0 if mask_ratio is None else mask_ratio
        n_masked = int(round(ratio * k))
        # Unmasked slots are filled with pad rather than real answer tokens: the probe
        # must never see the class name (§11 check 7).
        filler = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        row = [self.mask_token_id] * n_masked + [filler] * (k - n_masked)
        return torch.tensor([row] * batch, device=self.device, dtype=torch.long)

    def _forward_once(
        self,
        images: List[Image.Image],
        prompt: str,
        layers: Sequence[int],
        mask_ratio: Optional[float],
    ) -> Taps:
        input_ids, pixel_values, image_sizes = self._prepare(images, prompt)

        # --- T0 / T1: run the vision path ourselves so we see it pre-splice.
        raw = pixel_values if isinstance(pixel_values, list) else list(pixel_values)
        t0_list, t1_list, cls_list = [], [], []
        for tiles in raw:
            tiles = tiles if tiles.dim() == 4 else tiles[None]
            feats = self.vision_tower(tiles)            # [n_tiles, 729, d_vis], layer -2, patch
            t0_list.append(feats.reshape(-1, feats.shape[-1]))
            t1_list.append(self.projector(feats).reshape(-1, self.d_llm))
            cls_list.append(feats.mean(dim=(0, 1)))     # SigLIP2 has no CLS; mean stands in
        widths = {t.shape[0] for t in t0_list}
        if len(widths) > 1:
            # Ragged tiling within a batch. Pooling is per example, so the honest fix
            # is batch size 1 rather than padding a tap tensor we are about to average.
            raise RuntimeError(
                f"any-res tiling produced ragged patch counts {sorted(widths)} in one batch; "
                "run LLaDA-V extraction with --batch-size 1 or --llada-aspect pad"
            )
        t0 = torch.stack(t0_list)
        t1 = torch.stack(t1_list)
        t0_cls = torch.stack(cls_list)

        # --- splice, then run the LLM stack.
        answer_ids = self._answer_span_ids(len(images), mask_ratio)
        prep = self.model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, pixel_values, image_sizes
        )
        inputs_embeds = prep[4]
        attention_mask = prep[1]
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device, dtype=torch.long)

        spans = self._spans_from_length(input_ids, inputs_embeds.shape[1])

        if answer_ids is not None:
            embed_tokens = self.model.get_model().embed_tokens
            answer_embeds = embed_tokens(answer_ids).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, answer_embeds], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(answer_ids.shape, device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )

        # Call the bare transformer (LLaDAModel), not either wrapper forward.
        #
        # Neither wrapper is usable for tap extraction, for different reasons:
        #
        # * LlavaLLaDAModelLM.forward assigns `conversation_ids` only inside two
        #   branches both guarded on `inputs_embeds is None`, then forwards that
        #   variable unconditionally. Passing precomputed embeds — which a tap
        #   extractor must, the splice having already happened — skips both and
        #   raises UnboundLocalError. The fork only ever calls it with input_ids.
        #
        # * LLaDAModelLM.forward is a *training* forward: it unconditionally runs
        #   forward_process_embeds(), which injects random masking noise at a
        #   sampled timestep, and it requires `labels` to build prompt_index. Using
        #   it would make every hidden state a draw from the noising distribution
        #   rather than a deterministic function of the image — silently, since it
        #   returns perfectly well-shaped tensors.
        #
        # LLaDAModel is the plain stack: embeds in, hidden states out, no noise and
        # no loss. That is precisely §9's "clean single forward pass", and it is
        # what makes T0-T3 comparable with the LLaVA models. use_cache=False
        # because the model asserts KV caching is unsupported for MDM.
        out = self.model.get_model()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.hidden_states
        t3 = hidden[-1]
        t2 = torch.stack([t3[i, s:e] for i, (s, e) in enumerate(spans)])

        mid = {int(k): torch.stack([hidden[k][i, s:e] for i, (s, e) in enumerate(spans)])
               for k in layers}

        extra: Dict[str, Any] = {
            "mask_ratio": mask_ratio,
            "n_image_tokens": [e - s for s, e in spans],
            "n_patch_tokens": t0.shape[1],
        }
        if answer_ids is not None:
            # The tap that is actually on-policy for a diffusion decoder (§9).
            k = answer_ids.shape[1]
            extra["mask_states"] = t3[:, -k:]
            extra["mask_first"] = t3[:, -k]

        seq_len = attention_mask.sum(1).tolist()
        taps = Taps(
            t0=t0, t1=t1, t2=t2, t3=t3,
            img_span=spans, seq_len=seq_len, t0_cls=t0_cls, mid=mid, extra=extra,
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
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 32,
        steps: Optional[int] = None,
        block_length: Optional[int] = None,
    ) -> List[str]:
        """Diffusion decoding with the repo's defaults.

        ``steps`` and ``gen_length`` are recorded in the results row: they are not
        comparable to AR greedy decoding and the table must say so (§6.1).
        """
        input_ids, pixel_values, image_sizes = self._prepare(images, prompt)
        gen_length = max_new_tokens
        steps = steps or gen_length
        # generate_with_embeds asserts `gen_length % block_length == 0` and
        # defaults block_length to 128, so any gen_length under 128 fails outright.
        # Defaulting it to gen_length gives one block — full-length diffusion with
        # no semi-autoregressive remasking, which is the closest analogue to the
        # single greedy pass the LLaVA models get.
        block_length = block_length or gen_length
        assert gen_length % block_length == 0, (
            f"gen_length {gen_length} must be a multiple of block_length {block_length}"
        )
        self.last_decode_params = {
            "steps": steps, "gen_length": gen_length, "block_length": block_length,
            "remasking": "low_confidence", "temperature": 0.0,
        }
        out = self.model.generate(
            input_ids, images=pixel_values, image_sizes=image_sizes,
            gen_length=gen_length, steps=steps, block_length=block_length,
        )
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]

    def manifest(self) -> Dict[str, Any]:
        d = super().manifest()
        d.update(
            hf_id=self.hf_id,
            mode=self.mode,
            mask_k=self.mask_k,
            mask_ratios=list(self.mask_ratios),
            aspect_ratio=self.aspect_ratio,
            native_aspect_ratio=self.native_aspect_ratio,
            patch_merge_type=self.patch_merge_type,
            mask_token_id=self.mask_token_id,
            span_derivation="placeholder index + length delta, asserted",
            dtype=str(self.dtype),
        )
        return d
