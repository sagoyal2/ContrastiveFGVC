"""The one interface that matters (spec §10).

``TapExtractor.forward_taps`` runs a single forward pass per batch and returns
every representation tap defined in §5. Pooling to ``[d]`` happens in
``extract.py`` immediately afterwards — nothing per-token is ever written to disk
on the default path (§8, "the one rule that decides whether this is feasible").

Two things every subclass must get right, because they silently corrupt every
downstream number if wrong:

* the **image-token span** must be per-example (padding shifts it; §13), and
* ``t0`` must come from the ViT layer the model actually consumes
  (``mm_vision_select_layer``, normally ``-2``), patch tokens only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor

# Canonical tap names. Kept as a tuple so filenames, CLI parsing and the figure
# code all agree on spelling.
TAP_NAMES = ("T0c", "T0", "T1", "T2", "T3")
POOLINGS = ("avg", "final", "cls")


@dataclass
class Taps:
    """One batch's worth of taps. Shapes are per §5, batch-first.

    ``img_span`` is per example rather than a single tuple: under any-res tiling
    (LLaDA-V) the width varies, and under batched extraction padding shifts the
    offsets. Storing it per example is the only version that stays correct.
    """

    t0: Tensor                                   # [B, m, d_vis]  penultimate ViT, patch tokens
    t1: Tensor                                   # [B, m, d_llm]  post-projector (== S_o)
    t2: Tensor                                   # [B, m, d_llm]  last LLM layer, image span (== H_o)
    t3: Tensor                                   # [B, T, d_llm]  last LLM layer, full sequence
    img_span: List[Tuple[int, int]]              # per-example (start, end) into t3
    seq_len: List[int]                           # per-example count of real (non-pad) tokens
    t0c: Optional[Tensor] = None                 # [B, d_contrastive] L2-normed contrastive embedding
    t0_cls: Optional[Tensor] = None              # [B, d_vis] ViT CLS (LLaVA drops it; kept for reference)
    mid: Dict[int, Tensor] = field(default_factory=dict)  # layer_idx -> [B, m, d_llm]
    extra: Dict[str, Any] = field(default_factory=dict)   # model-specific (e.g. LLaDA mask states)

    def batch_size(self) -> int:
        return self.t3.shape[0]

    def validate(self) -> None:
        b = self.t3.shape[0]
        assert self.t0.shape[0] == b and self.t1.shape[0] == b and self.t2.shape[0] == b
        assert len(self.img_span) == b and len(self.seq_len) == b

        # T0 and T1 are the same tokens either side of a position-wise MLP, so their
        # counts must agree. T2's count may differ: LLaDA-V's any-res path inserts
        # newline tokens and drops padded tiles (`spatial_unpad`) between the
        # projector and the LLM sequence, so the spliced span is narrower than the
        # raw patch grid. Pooling is per-example, so this is harmless — but it must
        # not be silently assumed away.
        assert self.t0.shape[1] == self.t1.shape[1], (
            f"T0/T1 token counts disagree either side of a position-wise projector: "
            f"T0={self.t0.shape[1]} T1={self.t1.shape[1]}"
        )
        m2 = self.t2.shape[1]
        for i, (s, e) in enumerate(self.img_span):
            assert e - s == m2, f"example {i}: span width {e - s} != T2 width {m2}"
            assert 0 <= s < e <= self.t3.shape[1], f"example {i}: span {(s, e)} outside T3"


def pool_tokens(x: Tensor, how: str, valid_len: Optional[Sequence[int]] = None) -> Tensor:
    """Pool ``[B, T, d]`` to ``[B, d]``.

    ``valid_len`` gives the per-example count of real tokens, so that padded
    positions never enter an average and ``final`` picks the last *real* token
    rather than a pad. This is the §13 "padding in batched extraction" trap.
    """
    if x.dim() == 2:  # already pooled
        return x
    b, t, _ = x.shape
    if valid_len is None:
        valid_len = [t] * b
    lens = torch.as_tensor(list(valid_len), device=x.device)
    if how == "avg":
        mask = (torch.arange(t, device=x.device)[None, :] < lens[:, None]).to(x.dtype)
        return (x * mask[..., None]).sum(1) / mask.sum(1).clamp(min=1)[:, None]
    if how == "final":
        idx = (lens - 1).clamp(min=0)
        return x[torch.arange(b, device=x.device), idx]
    raise ValueError(f"unknown pooling {how!r} for a token sequence")


class TapExtractor(ABC):
    """Base class for every model under test (§4)."""

    name: str = "base"
    d_vis: int = 0
    d_llm: int = 0
    m: Optional[int] = None      # None means "variable, derive per example"
    n_layers: int = 0
    has_llm: bool = True

    @abstractmethod
    def forward_taps(
        self,
        images: List[Image.Image],
        prompt: str,
        layers: Sequence[int] = (),
    ) -> Taps:
        """One forward pass -> all taps for this batch."""

    def forward_text_taps(self, prompts: List[str]) -> Tuple[Tensor, List[int]]:
        """T3-txt: text-only control (§5, §6.3).

        Returns ``([B, T, d_llm], per-example real token count)``. Models without
        an LLM (the bare encoders) do not implement this.
        """
        raise NotImplementedError(f"{self.name} has no text-only tap")

    def generate(self, images: List[Image.Image], prompt: str, max_new_tokens: int = 32) -> List[str]:
        """Experiment 1 (§6.1). Greedy for AR models; diffusion default for LLaDA-V."""
        raise NotImplementedError(f"{self.name} does not implement generation")

    def manifest(self) -> Dict[str, Any]:
        """Provenance recorded next to every feature file (§8.4)."""
        return {
            "extractor": self.name,
            "d_vis": self.d_vis,
            "d_llm": self.d_llm,
            "m": self.m,
            "n_layers": self.n_layers,
        }
