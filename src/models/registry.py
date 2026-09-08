"""Config-driven model construction, so every CLI takes a plain ``--model`` name."""

from __future__ import annotations

from typing import Any, Dict

import torch
import yaml

from ..paths import MODEL_CONFIGS_DIR
from .base import TapExtractor

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model_config(name: str) -> Dict[str, Any]:
    path = MODEL_CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in MODEL_CONFIGS_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no model config {name!r}; available: {available}")
    cfg = yaml.safe_load(path.read_text())
    cfg["name"] = name
    return cfg


def list_models() -> list[str]:
    return sorted(p.stem for p in MODEL_CONFIGS_DIR.glob("*.yaml"))


def build_model(name: str, device: str = "cuda", **overrides: Any) -> TapExtractor:
    cfg = load_model_config(name)
    kind = cfg["kind"]
    # §8.5: bf16 everywhere. Mixing fp16 and bf16 across taps shifts probe numbers
    # by a few tenths and you will chase ghosts.
    dtype = _DTYPES[cfg.get("dtype", "bfloat16")]

    if kind == "llava":
        from .llava import LLaVATapExtractor

        model = LLaVATapExtractor(
            hf_id=cfg["hf_id"],
            device=device,
            dtype=dtype,
            expect_select_layer=cfg.get("mm_vision_select_layer", -2),
            expect_select_feature=cfg.get("mm_vision_select_feature", "patch"),
        )
    elif kind == "encoder":
        from .encoders import EncoderTapExtractor

        model = EncoderTapExtractor(
            hf_id=cfg["hf_id"],
            device=device,
            dtype=dtype,
            select_layer=cfg.get("select_layer", -2),
            has_cls=cfg.get("has_cls", True),
        )
    elif kind == "llada_v":
        from .llada_v import LLaDAVTapExtractor

        model = LLaDAVTapExtractor(
            hf_id=cfg["hf_id"],
            device=device,
            dtype=dtype,
            mode=overrides.get("llada_mode", cfg.get("mode", "clean")),
            mask_k=overrides.get("llada_k", cfg.get("mask_k", 8)),
            aspect_ratio=overrides.get("llada_aspect", cfg.get("aspect_ratio")),
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
        )
    else:
        raise ValueError(f"unknown model kind {kind!r} in {name}.yaml")

    model.config_name = name  # type: ignore[attr-defined]
    model.config_dict = cfg   # type: ignore[attr-defined]
    return model
