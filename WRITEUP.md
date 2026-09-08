# Where does concept–attribute signal get lost between the vision encoder and the LM?

**Phase 1 (CUB-200-2011) results.** Status: all four §0.5 gates pass; Phase 2 not started.
All numbers are single-seed (`seed 0`), `logreg` head (sklearn L-BFGS), CUB test split
(5,794 images, 200 classes). Figures: `figures/fig5_recreation.pdf`, `figures/layer_sweep.pdf`, `figures/llada_masked.pdf`. Raw:
`results/table2_probes.csv`, `results/table3_attributes.csv`. **Every measured number
appears in the appendix below**, generated from those CSVs by
`python -m src.report --appendix` so it cannot drift from them.

---

## 1. What was measured

Seven representation taps across five models, one forward pass per image, pooled before
anything reached disk. Class probes (200-way) and multi-label attribute probes (269 of
CUB's 312 binary attributes) were trained on the *same* cached features, so any
difference between them is a property of the representation and not of the setup.

| tap | where |
|---|---|
| T0-c | contrastive embedding (encoders only) |
| T0 | pre-projection, ViT layer −2, patch tokens |
| T1 | post-projector (`F_β(T0)`) |
| T2 | last LLM layer, image-token span |
| T3 | last LLM layer, full sequence |

## 2. Headline: the drop is real, and it happens twice

| model | T0.avg | T1.avg | T2.avg | T2.final | T3.avg | T3.final |
|---|---|---|---|---|---|---|
| CLIP-L/14-336 (encoder) | 79.38 | — | — | — | — | — |
| SigLIP2-so400m (encoder) | 84.83 | — | — | — | — | — |
| LLaVA-1.5-7B | 79.43 | 72.25 | 63.84 | 16.76 | 64.89 | **70.76** |
| LLaVA-1.5-13B | 79.43 | 71.83 | 61.87 | 16.26 | 62.27 | **69.33** |
| LLaDA-V-8B | 73.46 | 64.36 | 60.75 | 27.75 | 60.80 | 23.02 |

`figures/fig5_recreation.pdf` plots the five series §12 specifies — T0.avg, T1.avg,
T2.avg, **T2.final**, T3.final — so its fourth bar is T2.final, not T3.avg. Both
columns are given here so the figure and the table can be read against each other.

The `final` columns are not one quantity. **T2.final** is the last image token: under
LLaVA's causal attention it has attended over the whole image, but it is a single
patch and probes far below the mean (16.76 vs 63.84). **T3.final** is the `:` token
for the LLaVA models — the state the answer is decoded from, and the strongest tap
they have (70.76). For LLaDA-V neither reading applies: it decodes from
appended mask positions rather than from the last prompt token, so T3.final is just
some text position, which is why it (23.02) sits beside T3.pos0 (22.71) rather than
near T3.avg. **Do not compare LLaDA-V's T3.final against the LLaVA models' — they are
structurally different quantities.**

Two distinct losses, different in kind:

* **The projector costs ~9% relative** (7B: 79.43 → 72.25, −9.0%; 13B −9.6%; LLaDA-V
  −12.4%). One position-wise MLP, and the FINER Fig. 5 direction reproduces on all
  three models.
* **The LLM stack costs a further ~10%**, but *not* immediately. The layer sweep
  (7B, avg pooling) reads L0 72.25 → L8 **73.20** → L16 68.59 → L24 66.43 → L32 63.84:
  flat-to-slightly-up through layer 8, then monotone decay.

That shape is the answer to the question the sweep existed to settle (§8.3B). It is
**progressive discarding as the stack converges on language space**, not *lossy
projection the LLM never recovers from*. Endpoint-only measurement could not have
distinguished them.

**Scale does not help.** The 13B is behind the 7B at every post-projector tap despite
5120-dim states. Whatever the projector loses, more LLM does not recover.

## 3. Parts survive; the composition does not

Normalising each probe to its own model's T0 baseline:

| | class probe (relative) | attribute probe (relative lift) |
|---|---|---|
| T0 → T1 | −9.0% | **+1.4%** |
| T0 → T2 | −19.6% | −9.9% |
| T0 → T3.avg | −18.3% | −9.7% |

(LLaVA-7B; the 13B and LLaDA-V agree to within ~2 points on both columns.)

Absolute attribute lift over the 11.65 prior, all models — the ceiling is CLIP-L's
contrastive embedding at +26.00, so the honest range is 0 to ~26:

| model | T0.avg | T1.avg | T2.avg | T3.avg | T3.final | T3.pos0 |
|---|---|---|---|---|---|---|
| CLIP-L (T0c = +26.00) | +24.33 | — | — | — | — | — |
| SigLIP2 (T0c = +25.90) | +25.81 | — | — | — | — | — |
| LLaVA-1.5-7B | +24.34 | **+24.67** | +21.94 | +21.98 | +21.36 | **+0.01** |
| LLaVA-1.5-13B | +24.34 | +23.83 | +21.52 | +21.60 | +21.26 | **+0.00** |
| LLaDA-V (clean) | +24.22 | +23.22 | +22.08 | +22.05 | +15.63 | +15.73 |

`T3.pos0` at +0.01 and +0.00 is the negative control working exactly: on a vector that
cannot contain image information, the multi-label probe recovers precisely the marginal
attribute frequency and nothing more. LLaDA-V's +15.73 at the same tap is not a failure
— under bidirectional attention position 0 sees the image (§5).

Attribute information is **unchanged across the projector** and loses roughly half as
much as class identity across the whole stack. What degrades is not the visual content
but the *fine-grained composition* of it into a species identity — which is the
concept–attribute framing the study set out to test.

**Caveat, stated because the raw numbers overstate this.** mAP and top-1 are not on the
same scale, and the attribute probe has compressed dynamic range: every image tap sits
between +21 and +26 lift, against a ceiling of +26.00 (CLIP-L T0-c). The factor-of-two
claim above is the normalised one and is the only one that should be quoted.

## 4. The probe-vs-generation gap

| model | question | strict | lenient | %unmatched |
|---|---|---|---|---|
| LLaVA-7B | "What type of object…" (Zhang B.4) | 0.00 | 0.97 | 95.69 |
| LLaVA-7B | "What is the species of the bird…" | **0.00** | 4.47 | 31.17 |
| LLaVA-13B | "What is the species of the bird…" | **0.00** | 4.19 | 33.14 |
| CLIP-L zero-shot (CUB) | — | 63.12 | 63.12 | 0 |
| SigLIP2 zero-shot (CUB) | — | 78.62 | 78.62 | 0 |
| CLIP-L zero-shot (ImageNet-val) | — | 75.19 | 75.19 | 0 |

The ImageNet row is a **single-template** number (`"a photo of a {}."`), not the
80-prompt ensemble §6.1 specifies, which is why it sits ~1.4 points below the
published ensembled 76.6% for CLIP-L/14-336. Recorded in
`configs/datasets/imagenet_val.yaml`.

T3.final probes at **70.76%** — on the very state the answer is decoded from — while the
model names the correct species **zero times in 5,794 images**. The information is
present and linearly decodable at the decoding position; the decoder does not emit it.

Both questions are reported because the first one measures the wrong thing. Under
Zhang et al.'s ImageNet-written prompt, 5,747 of 5,794 generations said "a bird" and only
46 contained any species string — granularity mismatch, not recognition failure. The
species question moves unmatched from 96% to 31% and the model *tries*: "a sparrow"
(802×), "a seagull" (649×), "a hummingbird" (544×). It answers at folk-taxonomy level
while CUB labels at species level.

**Read `strict`, not `lenient`.** 3,318 of the 4.47% came from the sentence-embedding
rung at τ=0.65, where "a sparrow" is near-equidistant from ~20 CUB sparrow species and
the assignment is close to arbitrary. `lenient` largely measures the matcher's
tie-breaking. `strict = 0.00` is the trustworthy figure.

## 5. The diffusion model behaves as predicted architecturally

LLaDA-V's bidirectional attention produces a signature the causal models cannot:

| | LLaVA-7B | LLaVA-13B | LLaDA-V |
|---|---|---|---|
| T3.pos0 | 0.33 | 0.33 | **22.71** |
| T3.final | 70.76 | 69.33 | 23.02 |
| T2.avg − T2.final | 47.1 | 45.6 | **33.0** |

Under causal attention position 0 sees nothing and sits at chance — which is why it
works as a falsification control. Under bidirectional attention it attends over all 729
image tokens and carries real signal.

**Bidirectional does not mean positions are interchangeable**, and the data is emphatic
about this. LLaDA applies rotary embeddings, so every position is fully distinguished;
what bidirectional attention removes is only the *causal* asymmetry, i.e. the left-to-right
accumulation that makes LLaVA's `:` token a running summary. Positions still differ by a
lot:

| positions | probe |
|---|---|
| image span (T2.avg) | 60.75 |
| all positions (T3.avg) | 60.80 — 729 of ~806 tokens are image, so the mean tracks them |
| text position 0 (T3.pos0) | 22.71 |
| last text position (T3.final) | 23.02 |

**37.7 points** separate image from text positions. `pos0 ≈ final` is therefore a
statement about two comparable *text* positions — neither of which the model decodes
from — not about positions in general. The avg-vs-final gap is 13–14 points narrower
than the causal models', as §9 predicts.

### 5.1 Masked mode (k = 32): the mask positions are the answer

Clean mode understates LLaDA-V, and by a lot. Appending 32 `[MASK]` tokens — the §9
`masked` tap, which is the configuration the diffusion decoder actually runs — changes
every post-LLM number, on identical images and an identical prompt:

| tap | clean | masked (k=32) | Δ |
|---|---|---|---|
| T0.avg / T1.avg (pre-LLM) | 73.46 / 64.36 | 73.46 / 64.36 | **0.00** — control |
| T2.avg (image span) | 60.75 | 66.66 | **+5.91** |
| T2.final | 27.75 | 31.74 | +3.99 |
| T3.avg (prompt+image) | 60.80 | 65.81 | +5.01 |
| T3.final (last prompt token) | 23.02 | 33.59 | +10.57 |
| T3.pos0 | 22.71 | 34.00 | +11.29 |
| **Tmask.avg** (answer span) | — | **67.24** | — |
| Tmask.first / Tmask.last | — | 46.63 / 45.72 | — |

Three things follow.

**The mask span carries the most class information of any tap.** `Tmask.avg` = 67.24
exceeds even the image span, and sits only 8.5% below the pre-projection ceiling
(73.46). Measured where LLaDA-V actually decodes, its retention is comparable to
LLaVA-7B's T3.final of 70.76 — not the catastrophic 23.02 that clean mode suggests.

**Adding the answer span improves the image representations themselves.** T2.avg rises
5.91 points although T0 and T1 are bit-identical. Under bidirectional attention the
image positions attend *to* the masks, so the presence of an answer span reshapes what
they encode. This cannot happen in a causal model, where appended tokens lie to the
right and image positions can never see them. It is the sharpest VDLM-specific result
here.

**The information is distributed across the span, not concentrated.** Any single mask
position probes at ~46 (first 46.63, last 45.72, and equal to each other — parity
again), while the mean over 32 reaches 67.24. Different mask positions carry
complementary information. An autoregressive model concentrates the decision at one
next-token position; this one spreads it.

**What the mask positions add is composition, not content.** On the attribute probe the
answer span is *not* better than the image span — lift +22.07 versus +22.39 — while on
the class probe it is (67.24 versus 66.66, and versus 60.75 in clean mode). The masks
are not accumulating extra visual detail; they are composing attributes already present
into a class identity. That is the concept–attribute hypothesis appearing exactly where
the architecture predicts it should.

`figures/llada_masked.pdf` plots this comparison.

**Caveat.** `k` is a free parameter, only k=32 was run, and LLaVA has no analogue of
this tap — §9's stated trade-off. `Tmask` is its own row and must not be merged into a
T3 column. Features live in `features/lladav_8b_masked32/`; `T3.*` there excludes the
mask span so clean-vs-masked is a controlled comparison (`notes/decisions.md`).

**Two of the spec's checks needed rewriting, not the data.** The position-0 control and
the "T2.avg ≈ T2.final" test were both encoded as architecture-blind thresholds; both
are causal-specific. They now compare against the causal models rather than an absolute
epsilon. In both cases the data was right and the check was wrong.

## 6. Validation

| gate | result |
|---|---|
| §11 mechanical checks | PASS — span width = m, prompt ends on `:`, blanking pixels moves only the span, `F_β(T0)` = live layer-0 state |
| gate 2 — external anchor | PASS — CLIP-L ImageNet-val probe **81.90%** |
| gate 3 — FINER direction | PASS — T0 > T1 on all three VLMs |
| gate 4 — probe ≫ generation | PASS — 70.76 vs 0.00 |

Free consistency checks that also held: `mid.L0.avg` = `T1.avg` exactly (layer 0 *is* the
projected embedding); `mid.L32.avg` = `T2.avg` exactly; LLaVA-7B and 13B agree to the
digit at T0 (shared frozen tower); T3.pos0 = 0.33% against a 0.5% chance floor.

## 7. Known limitations

1. **Single seed.** `probe.yaml` specifies `seeds: [0,1,2]` with mean ± std. The logreg
   head is deterministic, so error bars would require the torch head, which was scoped
   out. No spread is available; do not treat these as ± 0.
2. **LLaDA-V's T0 is not comparable to standalone SigLIP2's.** `aspect_ratio: pad` was
   forced to fix m = 729; `expand2square` letterboxes CUB's wide images, leaving the bird
   at ~45% of its squashed pixel area. Same tower, 84.83 vs 73.46. Three independent
   signatures confirm it (T0, T1.final = 4.90, T1.final mAP lift = +3.25 — the last patch
   is padding). Cross-model comparisons here use *relative* loss from each model's own T0.
3. **Only k = 32 was run for masked mode**, and `k` is free. The clean-vs-masked gap
   (§5.1) is large enough that the choice matters; §9's `sweep` mode over mask ratios
   was not run.
4. **LLaDA-V has no generative number.** Its `generate_with_embeds` is batch-size-1
   (~4.4 s/image, ~7 h for the split) and our vicuna-style prompt is not its native conv
   template — it emits `"ASSassistant:"`. Not run.
5. **No dimensionality control.** Taps are 1024–5120 wide and a wider linear probe is
   strictly more expressive. This does not threaten the headline — T0 (1024) *beats* T1
   and T2 (4096), so the drop runs against the width advantage — but comparisons that run
   the other way (T3.final at 4096 vs T0.avg at 1024) are unresolved.
6. **Scoped out by decision** (`notes/decisions.md`): T3-txt text control, further layer
   sweeps, PCA-1024, k-NN. ImageNet zero-shot used one template, not the 80-prompt
   ensemble.

## 8. What would sharpen this next

* **ImageNet-val, LLaVA-7B (~1 h).** The only external anchor *past the projector*:
  Zhang et al. report 77.1% at T3.final. Everything downstream of T1 is currently
  unanchored. Also gives the coarse-vs-fine contrast CUB alone cannot.
* **Re-extract SigLIP2 through `expand2square` (~3 min)** to confirm preprocessing fully
  explains the 11.4-point LLaDA-V T0 gap.
* **Phase 2** — Cars, Dogs, Aircraft — to test whether the CUB shape generalises across
  fine-grained domains.

---

# Appendix — complete results

Generated by `python -m src.report --appendix`; every value is read from `results/table2_probes.csv` and `results/table3_attributes.csv`. Single seed (0), `logreg` head, CUB test split unless noted.

## A1. Class probe, top-1 (CUB, 200 classes, chance 0.5%)

| model              |    T0c |   T0.avg |   T0.cls |   T1.avg |   T1.final |   T2.avg |   T2.final |   T3.avg |   T3.final |   T3.pos0 |   Tmask.avg |   Tmask.first |   Tmask.last |
|:-------------------|-------:|---------:|---------:|---------:|-----------:|---------:|-----------:|---------:|-----------:|----------:|------------:|--------------:|-------------:|
| clip_l14_336       |  86.62 |    79.38 |    83.85 |   nan    |     nan    |   nan    |     nan    |   nan    |     nan    |    nan    |      nan    |        nan    |       nan    |
| siglip2_so400m     |  88.49 |    84.83 |   nan    |   nan    |     nan    |   nan    |     nan    |   nan    |     nan    |    nan    |      nan    |        nan    |       nan    |
| llava15_7b         | nan    |    79.43 |    84.54 |    72.25 |      25.42 |    63.84 |      16.76 |    64.89 |      70.76 |      0.33 |      nan    |        nan    |       nan    |
| llava15_13b        | nan    |    79.43 |    84.54 |    71.83 |      25.39 |    61.87 |      16.26 |    62.27 |      69.33 |      0.33 |      nan    |        nan    |       nan    |
| lladav_8b          | nan    |    73.46 |    73.44 |    64.36 |       4.9  |    60.75 |      27.75 |    60.8  |      23.02 |     22.71 |      nan    |        nan    |       nan    |
| lladav_8b_masked32 | nan    |    73.46 |    73.44 |    64.36 |       4.9  |    66.66 |      31.74 |    65.81 |      33.59 |     34    |       67.24 |         46.63 |        45.72 |

`T0c` is the contrastive embedding and exists only for the bare encoders. `T0.cls` is the ViT CLS token, which LLaVA drops; for SigLIP2 there is no CLS position at all, and for LLaDA-V the value shown is the patch mean. `Tmask.*` exists only in masked mode.

## A2. Class probe, top-5

| model              |    T0c |   T0.avg |   T0.cls |   T1.avg |   T1.final |   T2.avg |   T2.final |   T3.avg |   T3.final |   T3.pos0 |   Tmask.avg |   Tmask.first |   Tmask.last |
|:-------------------|-------:|---------:|---------:|---------:|-----------:|---------:|-----------:|---------:|-----------:|----------:|------------:|--------------:|-------------:|
| clip_l14_336       |  97.86 |    95.1  |    97.36 |   nan    |     nan    |   nan    |     nan    |   nan    |     nan    |    nan    |      nan    |        nan    |       nan    |
| siglip2_so400m     |  98.29 |    97.86 |   nan    |   nan    |     nan    |   nan    |     nan    |   nan    |     nan    |    nan    |      nan    |        nan    |       nan    |
| llava15_7b         | nan    |    95.27 |    97.26 |    92.06 |      50.31 |    87.4  |      37.23 |    87.5  |      91.7  |      2.59 |      nan    |        nan    |       nan    |
| llava15_13b        | nan    |    95.27 |    97.26 |    92.3  |      50.62 |    85.93 |      36.45 |    86.16 |      91.04 |      2.59 |      nan    |        nan    |       nan    |
| lladav_8b          | nan    |    93.29 |    93.29 |    88.87 |      15.22 |    88.52 |      58.27 |    88.42 |      51.69 |     51.93 |      nan    |        nan    |       nan    |
| lladav_8b_masked32 | nan    |    93.29 |    93.29 |    88.87 |      15.22 |    90.58 |      61.74 |    91.03 |      67.14 |     65.77 |       90.73 |         78.08 |        78.46 |

## A3. Layer sweep (LLaVA-1.5-7B, CUB) — both poolings

Layer 0 is the projected image embedding spliced into the sequence; layer 32 is the last LLM layer. `mid.L0.avg` = `T1.avg` and `mid.L32.avg` = `T2.avg` exactly, which is the free consistency check.

|   layer |   avg |   final |
|--------:|------:|--------:|
|       0 | 72.25 |   25.42 |
|       8 | 73.2  |   25.18 |
|      16 | 68.59 |   18.29 |
|      24 | 66.43 |   15.14 |
|      32 | 63.84 |   16.76 |

## A4. Attribute probe — mean AP over 269 CUB attributes

Prior (a probe that has learned nothing) = **11.65**. Read the lift in A5, not the raw mAP.

| model              |    T0c |   T0.avg |   T0.cls |   T1.avg |   T1.final |   T2.avg |   T2.final |   T3.avg |   T3.final |   T3.pos0 |   Tmask.avg |   Tmask.first |   Tmask.last |
|:-------------------|-------:|---------:|---------:|---------:|-----------:|---------:|-----------:|---------:|-----------:|----------:|------------:|--------------:|-------------:|
| clip_l14_336       |  37.64 |    35.98 |    36.59 |   nan    |     nan    |   nan    |     nan    |   nan    |     nan    |    nan    |      nan    |        nan    |       nan    |
| siglip2_so400m     |  37.54 |    37.46 |   nan    |   nan    |     nan    |   nan    |     nan    |   nan    |     nan    |    nan    |      nan    |        nan    |       nan    |
| llava15_7b         | nan    |    35.99 |    36.59 |    36.32 |      22.51 |    33.58 |      18.72 |    33.62 |      33    |     11.66 |      nan    |        nan    |       nan    |
| llava15_13b        | nan    |    35.99 |    36.59 |    35.48 |      22.64 |    33.16 |      17.92 |    33.24 |      32.91 |     11.65 |      nan    |        nan    |       nan    |
| lladav_8b          | nan    |    35.86 |    35.86 |    34.87 |      14.9  |    33.73 |      26.04 |    33.7  |      27.28 |     27.38 |      nan    |        nan    |       nan    |
| lladav_8b_masked32 | nan    |    35.86 |    35.86 |    34.87 |      14.9  |    34.04 |      26.84 |    34.03 |      29.5  |     30.16 |       33.71 |         30.91 |        30.58 |

## A5. Attribute probe — lift over prior

The honest range for this experiment is 0 to ~26; the ceiling is CLIP-L's contrastive embedding. `T3.pos0` at +0.01 is the negative control: on a constant vector the probe recovers exactly the marginal.

| model              |   T0c |   T0.avg |   T0.cls |   T1.avg |   T1.final |   T2.avg |   T2.final |   T3.avg |   T3.final |   T3.pos0 |   Tmask.avg |   Tmask.first |   Tmask.last |
|:-------------------|------:|---------:|---------:|---------:|-----------:|---------:|-----------:|---------:|-----------:|----------:|------------:|--------------:|-------------:|
| clip_l14_336       |  26   |    24.33 |    24.94 |   nan    |     nan    |   nan    |     nan    |   nan    |     nan    |    nan    |      nan    |        nan    |       nan    |
| siglip2_so400m     |  25.9 |    25.81 |   nan    |   nan    |     nan    |   nan    |     nan    |   nan    |     nan    |    nan    |      nan    |        nan    |       nan    |
| llava15_7b         | nan   |    24.34 |    24.94 |    24.67 |      10.87 |    21.94 |       7.07 |    21.98 |      21.36 |      0.01 |      nan    |        nan    |       nan    |
| llava15_13b        | nan   |    24.34 |    24.94 |    23.83 |      10.99 |    21.52 |       6.27 |    21.6  |      21.26 |      0    |      nan    |        nan    |       nan    |
| lladav_8b          | nan   |    24.22 |    24.22 |    23.22 |       3.25 |    22.08 |      14.39 |    22.05 |      15.63 |     15.73 |      nan    |        nan    |       nan    |
| lladav_8b_masked32 | nan   |    24.22 |    24.22 |    23.22 |       3.25 |    22.39 |      15.19 |    22.38 |      17.86 |     18.52 |       22.07 |         19.26 |        18.93 |

## A6. ImageNet-val (1000 classes, 40k train / 10k test carve)

| model        | dataset      | key   |   d |   top1 |   top5 |
|:-------------|:-------------|:------|----:|-------:|-------:|
| clip_l14_336 | imagenet_val | T0c   | 768 |   81.9 |  96.58 |
