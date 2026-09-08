"""Prompts (§5.2, §6.1, §6.3). Every string that reaches a model lives here.

Two distinct prompt families, and they must not be mixed up:

* ``PROBE_PROMPT`` — the fixed Zhang et al. B.4 prompt used for **all** taps in
  Experiment 2. It never varies between taps, and it never names the class
  (§11 check 7, label leakage).
* zero-shot templates — encoder-only, Experiment 1's T0-c row.

The no-image text control (spec §6.3's T3-txt) is deliberately not implemented —
see notes/decisions.md.
"""

from __future__ import annotations

from typing import List

CHAT_PREAMBLE = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions. "
)

# Zhang et al. B.4's question. Written for ImageNet, where "object" is the right
# granularity — which is exactly why it needs a companion on fine-grained sets.
DEFAULT_QUESTION = "What type of object is in this photo?"


def generative_prompt(question: str) -> str:
    """Wrap a question in the vicuna_v1 chat frame with an image placeholder.

    The trailing ':' carries no space after it — that ':' is the token T3.final
    pools, and a trailing space would tokenise differently.
    """
    return f"{CHAT_PREAMBLE}USER: <image>\n{question} ASSISTANT:"


# §5.2, verbatim. **Fixed for every tap and never varied** — the cached features
# are tied to this exact string via the manifest's prompt_sha1. Experiment 1 may
# swap the question (see `generative_prompt`); Experiment 2 may not.
PROBE_PROMPT = generative_prompt(DEFAULT_QUESTION)

# Same wording, no image — the LLaVA processor would otherwise expand nothing.
PROBE_PROMPT_NO_IMAGE = PROBE_PROMPT.replace("<image>\n", "")

DEFAULT_ZS_TEMPLATES = ["a photo of a {}."]

def zeroshot_prompts(classnames: List[str], templates: List[str]) -> List[List[str]]:
    """``[n_classes][n_templates]`` prompts, for the ensemble average (§6.1)."""
    return [[t.format(c) for t in templates] for c in classnames]
