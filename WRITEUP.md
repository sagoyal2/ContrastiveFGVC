# Where does concept–attribute signal get lost between the vision encoder and the LM?

**Phase 1 (CUB-200-2011) results.** Status: all four §0.5 gates pass; Phase 2 not started.
All numbers are single-seed (`seed 0`), `logreg` head (sklearn L-BFGS), CUB test split
(5,794 images, 200 classes). Figures: `figures/fig5_recreation.pdf`,
`figures/layer_sweep.pdf`. Raw: `results/table2_probes.csv`, `results/table3_attributes.csv`.

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

| model | T0.avg | T1.avg | T2.avg | T3.avg | T3.final |
|---|---|---|---|---|---|
| CLIP-L/14-336 (encoder) | 79.38 | — | — | — | — |
| SigLIP2-so400m (encoder) | 84.83 | — | — | — | — |
| LLaVA-1.5-7B | 79.43 | 72.25 | 63.84 | 64.89 | **70.76** |
| LLaVA-1.5-13B | 79.43 | 71.83 | 61.87 | 62.27 | **69.33** |
| LLaDA-V-8B | 73.46 | 64.36 | 60.75 | 60.80 | 23.02 |

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
| CLIP-L zero-shot | — | 63.12 | 63.12 | 0 |
| SigLIP2 zero-shot | — | 78.62 | 78.62 | 0 |

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
image tokens and carries real signal, and `pos0 ≈ final` (22.71 vs 23.02) because no
position is privileged. The avg-vs-final gap is 13–14 points narrower than the causal
models', as §9 predicts.

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
3. **LLaDA-V has no generative number.** Its `generate_with_embeds` is batch-size-1
   (~4.4 s/image, ~7 h for the split) and our vicuna-style prompt is not its native conv
   template — it emits `"ASSassistant:"`. Not run.
4. **No dimensionality control.** Taps are 1024–5120 wide and a wider linear probe is
   strictly more expressive. This does not threaten the headline — T0 (1024) *beats* T1
   and T2 (4096), so the drop runs against the width advantage — but comparisons that run
   the other way (T3.final at 4096 vs T0.avg at 1024) are unresolved.
5. **Scoped out by decision** (`notes/decisions.md`): T3-txt text control, further layer
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
