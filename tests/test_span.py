"""§11 check 5 — the image-token span. These are the checks that, if wrong,
silently corrupt every downstream number rather than raising.

Marked ``gpu``/``data``: they need real weights and real images.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.registry import get_dataset
from src.data.templates import PROBE_PROMPT
from src.models.registry import build_model
from src.paths import use_storage_hf_cache

pytestmark = [pytest.mark.gpu, pytest.mark.data]
N_IMAGES = 20


@pytest.fixture(scope="module")
def model():
    use_storage_hf_cache()
    return build_model("llava15_7b")


@pytest.fixture(scope="module")
def images():
    ds = get_dataset("cub", "test")
    return [ds[i][0] for i in range(N_IMAGES)]


def test_span_width_equals_m(model, images):
    taps = model.forward_taps(images[:8], PROBE_PROMPT)
    for s, e in taps.img_span:
        assert e - s == model.m == 576


def test_prompt_ends_on_colon(model):
    """§5.2: T3.final pools the ':' token, so it must actually be the last token."""
    ids = model.tokenizer(PROBE_PROMPT.replace("<image>\n", ""))["input_ids"]
    assert model.tokenizer.decode(ids[-1:]).strip() == ":", (
        f"last token is {model.tokenizer.decode(ids[-1:])!r}, not ':'"
    )


def test_blanking_pixels_changes_only_the_image_span(model, images):
    """Zeroing pixels must move ``t3`` inside the span and not before it.

    'Not before it' holds only because attention is causal: positions left of the
    span cannot see the image. It is the sharpest available proof that the span
    indices point at the image tokens and not at some offset neighbourhood.
    """
    batch = images[:4]
    taps = model.forward_taps(batch, PROBE_PROMPT)
    s, e = taps.img_span[0]
    t3_ref = taps.t3.float().clone()

    black = [Image.new("RGB", im.size) for im in batch]
    taps_black = model.forward_taps(black, PROBE_PROMPT)
    t3_black = taps_black.t3.float()

    assert taps_black.img_span[0] == (s, e)
    inside = (t3_ref[:, s:e] - t3_black[:, s:e]).abs().max().item()
    before = (t3_ref[:, :s] - t3_black[:, :s]).abs().max().item()
    assert inside > 1e-2, f"blanking the image did not change the span (max diff {inside})"
    assert before < 1e-2, f"positions before the span moved (max diff {before}) — span is wrong"


def test_t1_identity(model, images):
    """§11 check 4: ``projector(T0)`` standalone == the live layer-0 hidden state
    at the image positions. Catches a select_layer / select_feature mismatch,
    which is otherwise invisible."""
    taps = model.forward_taps(images[:4], PROBE_PROMPT, want_t1_live=True)
    live = taps.extra["t1_live"].float()
    standalone = taps.t1.float()
    rel = (live - standalone).abs().max().item() / standalone.abs().max().item()
    assert rel < 0.02, f"T1 mismatch: relative max diff {rel:.4f}"
