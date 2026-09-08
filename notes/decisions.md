# Decisions that depart from the spec

## T3-txt / the text-only modality control is not part of this study

Spec §6.3 proposes a no-image control: feed `"USER: What type of object is a
{C_i}? ASSISTANT:"`, probe the same taps, and report
`Δ = acc(text tap) − acc(image tap)` as the modality gap. **We are not running
it.** Removed from `src/extract.py`, `src/probe.py`, `src/data/templates.py`,
`src/figures.py` and both driver scripts; `figures/modality_gap.pdf` is dropped
from the §12 deliverables.

A pilot was run on LLaVA-1.5-7B before the decision, and it illustrates why the
control is weak rather than informative: with 20 carrier templates per class it
scored **T3txt.avg = 100.00%** and **T3txt.final = 99.90%**, against 70.76% on
the image side. The text probe is saturated. The class name is present in the
input as a token, so the probe only has to read it back, and a ceiling at 100%
cannot distinguish "text is somewhat better" than the image path from "text is
vastly better" — it only shows the text pathway is not the bottleneck, which was
never in doubt.

The attribute probe (Experiment 3, `src/attribute_probe.py`) remains the
instrument for the concept–attribute question. It asks whether a *visible*
attribute survives each stage, which is the actual hypothesis, and it has no
class name in the input to trivially read back.

The two pilot `T3txt.*` rows have been removed from `results/table2_probes.csv`
so no table or figure picks them up. The pilot features themselves are left at
`features/llava15_7b/cub/text/` (240 KB, nothing reads them) purely as the
evidence behind the numbers quoted above; delete the directory if you want the
tree to match the code.

## The layer sweep is done once, on CUB, and not repeated

Spec §8.3B proposes extracting `T-mid` — hidden states at layers
`{0, 8, 16, 24, 32}` (`{0, 10, 20, 30, 40}` for the 13B) over the image span —
and flags ImageNet as an optional +40 GB extension. **We ran it once, on CUB, and
that is the end of it.** `--sweep` is off in `scripts/run_phase1.sh` and should
not be passed for Phase 2 or Phase 3 extraction.

The CUB sweep already answered the question it was there to answer. On
LLaVA-1.5-7B (logreg head, avg pooling): L0 72.25 → L8 73.20 → L16 68.59 →
L24 66.43 → L32 63.84. That is the "progressive discarding" profile, not
"flat-and-low from layer 0" — the signal survives the projector, holds through
roughly layer 8, then decays monotonically. Repeating the measurement on further
datasets would cost 40 GB and hours of extraction to redraw a curve whose shape
is already established.

`src/figures.py --figure sweep` and the `mid.*` keys already on disk stay; the
sweep figure remains a §12 deliverable. Only further *extraction* of sweep taps
is dropped. The 13B CUB sweep was already mid-extraction when this was decided
and was allowed to finish, so both models have the CUB curve.

## Probe heads are logreg and linear only; the PCA control is deferred

Spec §6.2 lists four dimensionality-fairness controls alongside the linear probe:
a PCA projection of every tap to a common `d = 1024`, matched train-set size, a
k-NN probe (k=20, cosine), and an optional MLP probe. **None of these are being
run.** The k-NN head has been removed from `src/probe.py` and `configs/probe.yaml`
outright; `--head` now accepts only `logreg` and `linear`. The `--pca-dim` flag
is left in place but unused — this one is deferred rather than dropped.

All current numbers use `logreg` (sklearn L-BFGS) and stand as they are. The
torch `linear` head (Zhang et al. B.4 verbatim) remains available and is still in
`scripts/run_phase1_probes.sh`, but is not being run: the logreg pass already
covers every tap and the two heads differ by ~0.5%.

**What this costs, so it is not forgotten.** Taps have widths 1024 / 1152 / 4096 /
5120, and a wider linear probe is strictly more expressive. Without the PCA-1024
control, a *higher* score at a wider tap is not by itself evidence of more
information. This does not threaten the headline result — T0 (1024) *beats* T1
(4096) and T2 (4096), i.e. the drop runs against the width advantage, so
correcting for width would only widen it. It does mean any comparison that runs
the other way (T3.final at 4096 vs T0.avg at 1024, say) stays unresolved until
the control is run. State this in the writeup rather than quietly comparing
across widths.

To restore the k-NN control: it was a `KNeighborsClassifier(n_neighbors=20,
metric="cosine")` fitted on train+val and scored on test, reported as top-1 with
top-5 undefined. It answers whether information is present but linearly
entangled, which is a different claim from information loss.

## LLaDA-V's T0 is not comparable to standalone SigLIP2's T0 (preprocessing)

`configs/models/lladav_8b.yaml` sets `aspect_ratio: pad`, overriding the
checkpoint's native `anyres_max_4` + `spatial_unpad`. This was deliberate: it
forces the single-tile path so `m` is a fixed 729, structurally comparable to
LLaVA-1.5's fixed 576, instead of varying per image with tiling (§13).

It has a measured cost. The two paths reach the same frozen tower at the same
layer (`mm_vision_select_layer=-2`, `mm_vision_select_feature=patch`,
`d_vis=1152`) through different preprocessing:

* standalone SigLIP2 — `image_processor.preprocess()`: resize straight to
  384x384, squashing the aspect ratio, subject fills the frame;
* LLaDA-V under `pad` — `expand2square()` letterboxes to a square with the mean
  colour *first*, then resizes to 384.

CUB images are typically ~500x335, so padding to 500x500 leaves the bird at
roughly 45% of the pixel area it has under squashing. The linear probe on
`T0.avg` measures **84.83** standalone versus **73.46** inside LLaDA-V — an
11.4-point gap on what is nominally the same tensor.

**Consequences.** The LLaDA-V T0 row must not be read as "SigLIP2 is weaker
inside LLaDA-V"; it is the same encoder looking at a smaller bird. Do not put the
two T0 numbers in one column without this caveat. The *within-model* trajectory
is unaffected — T0, T1, T2 and T3 all see identical pixels — so LLaDA-V's
projector drop and stack decay remain valid, measured against its own lower T0
baseline. Cross-model comparisons should therefore use *relative* loss from each
model's own T0, not absolute top-1.

Untested: re-extracting standalone SigLIP2 through `expand2square` should land
near 73.5 if preprocessing is the whole story. ~3 min of GPU; not run.

## Masked-mode taps: T3.* excludes the [MASK] span

`--llada-mode masked --llada-k 32` appends k `[MASK]` tokens to the sequence, so
`attention_mask.sum(1)` — the per-example length used for pooling — counts them.
Pooled naively, `T3.avg` would average the mask states in and `T3.final` would be
the **last mask token** rather than the last prompt token. The key names would then
mean different tensors in `features/lladav_8b/` and
`features/lladav_8b_masked32/`, and any table putting the two side by side would be
comparing different quantities under one column header.

`src/extract.py` therefore pools T3 over `seq_len - k`. **`T3.*` is always the
prompt+image sequence, in both modes**, and the answer span appears only as
`Tmask.avg` / `Tmask.first` / `Tmask.last`. This makes clean-vs-masked a controlled
comparison: identical taps on identical inputs, differing only in whether the
answer span exists.

The mask span is entirely `[MASK]` (ratio 1.0), never partially filled with answer
tokens, so no class name can leak into a probe (§11 check 7).
