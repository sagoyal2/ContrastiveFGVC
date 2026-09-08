# Experiment Spec: Where does concept–attribute signal get lost between the vision encoder and the LM?

**Status:** design doc / handoff brief. Paste into an IDE agent session as the standing context for implementation.
**Owner:** Avi
**Last updated:** 2026-09-08

---

## 0. TL;DR for an implementing agent

**Start with Phase 1 in §0.5 — CUB only, ~26 min, ~5 GB. Do not extract all five datasets on the first run.**

You are building a **feature-tap + linear-probe harness** that measures how much class-discriminative information survives at each stage of a VLM/VDLM forward pass, for one coarse-grained and four fine-grained classification datasets, on LLaVA-1.5-7B, LLaVA-1.5-13B, and LLaDA-V-8B.

The end product is:
1. A table of **generative (open-ended) top-1 accuracy** per model per dataset.
2. A **grouped bar chart** (a recreation and extension of Figure 5 of Kim & Ji 2024) of **linear-probe top-1 accuracy** at each representation tap, per dataset.
3. A short writeup of where the drop happens.

Do **not** start by writing a training loop. Start by writing the *tap extractor* and validating it with the mechanical checks in §11 (which are dataset-agnostic and run on CUB alone).

---

## 0.5 Phased execution plan — **start here**

Do **not** extract all five datasets on the first run. ImageNet is 150k of the 208.5k total images and dominates both disk and wall clock, while being the least informative dataset for the hypothesis. Run CUB end-to-end first, confirm the pipeline and the effect, then scale.

### Phase 1 — CUB only (~26 min extraction, ~5 GB)

| | |
|---|---|
| Datasets | **CUB-200-2011 only** (5,994 train / 5,794 test) |
| Models | all three + both encoders |
| Taps | T0-c, T0, T1, T2, T3, T3-txt, **and** the CUB layer sweep |
| Experiments | 1 (generative), 2 (probes), **3 (attribute probe — CUB's 312 attributes are already aligned)** |
| Disk | 2.0 GB pooled + 2.9 GB sweep = **5.0 GB** |
| Wall clock | ~4 min (7B) + ~9 min (13B) + ~4 min (LLaDA-V) + load time ≈ **26 min** |
| Extra | **CLIP-L zero-shot + probe on ImageNet val only** (~2 min, no LLM passes) — see below |

**Why CUB is the right pilot:** smallest of the five; ships 312 binary attribute annotations so Experiment 3 runs immediately; appears in FINER Fig. 5 so the T0>T1 direction check applies; and it is fine-grained, i.e. where the hypothesized effect should be *largest*. A null result on CUB is itself informative and saves you the full run.

**What Phase 1 cannot check.** Sanity check #1 (Zhang et al.'s 77.1% LLaVA / 85.2% CLIP-L on ImageNet) is the only external reference number and it needs ImageNet. Mitigations:
- The *structural* bugs — wrong image-token span, wrong ViT layer, padding shift — are all caught by **dataset-agnostic** checks that run on CUB: span assert, T1 identity check, position-0 collapse, LLaDA bidirectionality. See §11 checks 2, 4, 5, 6. These are the failure modes that would silently corrupt every downstream number.
- Run **CLIP-L T0-c on ImageNet val only** in Phase 1 (encoder forward passes only, no LLM, ~2 min, ~0.2 GB). Expect the ~85% linear-probe reference. This validates the probe trainer and the encoder path before you trust anything past the projector.

**Phase 1 gate — proceed only if all hold:**
1. All §11 mechanical checks pass.
2. CLIP-L ImageNet-val probe lands near the published reference.
3. T0.avg > T1.avg on CUB (FINER's direction).
4. T3.final probe is far above generative top-1 on CUB (the probe-vs-generation gap exists).

If (3) or (4) fail, stop and debug — do **not** spend the remaining 17.7x of extraction.

### Phase 2 — remaining fine-grained sets (~1.5 h, ~11 GB)

Add Stanford Cars, Stanford Dogs, FGVC-Aircraft (46,765 images). This completes the fine-grained half of the FINER figure and tells you whether the CUB result generalizes across fine-grained domains. Cheap; run it as soon as Phase 1 gates.

### Phase 3 — ImageNet, coarse granularity (~4 h, ~20 GB)

Add the 100/class ImageNet subsample + val. This is the coarse-vs-fine contrast in your question 1, and it is where Zhang et al.'s reference numbers become directly comparable. Decide the layer-sweep-on-ImageNet question (+40 GB) only after seeing the CUB sweep curve.

### Phase 4 — optional extras

Per-token re-pooling cache (§8.3C), LLaDA-V mask-ratio sweep (§9), NABirds + iNaturalist for exact six-bar FINER parity.

> **Design implication for the implementer:** every CLI is already dataset-scoped (`--dataset cub`), and the `features/` tree is keyed by dataset, so phases are purely a matter of which commands you run. Nothing needs restructuring between phases. Write `figures.py` to read whatever datasets exist on disk and plot those, rather than assuming all five.

---

## 1. Hypothesis under test

> In V-LLMs and V-DLMs, the **vision input modality does not provide enough concept–attribute knowledge-retrieval signal** compared to the textual modality.

Operationalized: if you take the same concept (e.g. "Sooty Albatross") and present it (a) as an image and (b) as text, the representation the LLM ends up with is far more linearly class-separable in case (b). Furthermore, the degradation is **not** uniform across the stack — we want to localize it to a stage:

```
image → [vision encoder] → [projector] → [LLM layers] → answer
                        ↑             ↑              ↑
                       T0            T1          T2 / T3
```

The competing explanations we are trying to discriminate between:
- **H1 (projector bottleneck):** the modality connector destroys attribute information. Predicts a large T0→T1 drop. This is what Kim & Ji's Figure 5 reports.
- **H2 (LLM ignores it):** information survives projection but the LM's causal stack does not route it into the answer position. Predicts small T0→T1 drop, large T2→T3 drop, or a large probe-vs-generation gap.
- **H3 (it's all there, decoding is the problem):** probes stay high everywhere and only generation collapses. This is Zhang et al.'s conclusion (data/decoding-limited, not representation-limited) — LLaVA-1.5-7B probes at 77.1% on ImageNet vs CLIP-L's 85.2%, while generating far worse.

These are not mutually exclusive and the interesting result is the *shape of the curve across taps*, not any one number.

---

## 2. Source papers and exactly what we take from each

| Paper | arXiv | What we take |
|---|---|---|
| **FINER** — Kim & Ji, *Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in LVLMs* | [2402.16315](https://arxiv.org/abs/2402.16315) | **Figure 5** is the target artifact. Caption: *"Linear Probing on Projected Image Embeddings. Classification accuracy (%) for before and after image embedding projection to textual space."* Bar chart over 6 FGVC sets (iNaturalist, CUB-200-2011, FGVC-Aircraft, Stanford Dogs, NABirds, Stanford Cars), comparing CLIP-ViT-L/14 features **before** vs **after** LLaVA-1.5's projector. They report a consistent, substantial drop. We reproduce this and then **extend it past the projector into the LLM stack**. |
| **Zhang et al.** — *Why are Visually-Grounded Language Models Bad at Image Classification?* (NeurIPS 2024) | [2405.18415](https://arxiv.org/abs/2405.18415) | **Appendix B.4** gives the probing protocol we adopt verbatim where possible. See §6.2. Key points: features from the **final** LLM layer; prompt `USER: <576 Image Tokens> What type of object is in this photo? ASSISTANT:`; probe either **the last token (the `:`)** or **the average over all tokens**; probing *other* token positions collapses (Flowers102: 96.2% avg-token vs 0.7% at position 0 — Table 10), which is why only these two poolings are defensible. Probe hyperparameters: batch 512, lr 1e-3, Adam, 500 epochs, select by best val. |
| **Finedefics** | [2501.15140](https://arxiv.org/abs/2501.15140) | **Section 2 notation**, adopted wholesale below. It is the cleanest available notation for separating the visual object token sequence from the category text token sequence, and for the "final token vs sequence average" pooling choice. |

---

## 3. Notation (adopted from Finedefics §2 — use these symbols in code and in the writeup)

For image `I_i` containing object `O_i` with ground-truth category name `C_i`:

| Symbol | Meaning | Concretely |
|---|---|---|
| `V_α` | vision encoder | CLIP-ViT-L/14-336 (LLaVA-1.5) or SigLIP2-so400m-p14-384 (LLaDA-V) |
| `F_β` | learnable modality connector / projector | `mlp2x_gelu` for LLaVA-1.5 |
| `E_φ` | LLM token embedding layer | Llama-2/Vicuna embed for LLaVA; LLaDA-8B embed for LLaDA-V |
| `L_θ` | LLM transformer layers | 32 layers (7B/8B), 40 layers (13B) |
| `S_o^i = [o_1 … o_m]` | **visual object token sequence**, length `m` | `F_β(V_α(I_i))`. m=576 for LLaVA-1.5 |
| `S_c^i = [c_1 … c_n]` | **category text token sequence**, length `n` | `E_φ(tokenize(C_i))` |
| `H_o^i = L_θ(S_o^i)` | LLM output states **at image-token positions** | tap T2 |
| `H_c^i = L_θ(S_c^i)` | LLM output states **at text-token positions** | tap T3 / text control |
| `pool(·)` | global representation extractor | either `mean` over the sequence, or the **final** token of the span |

Everywhere below, "**avg**" = mean over the span's tokens; "**final**" = the last token of the span. Report both, always, as separate rows — they behave differently and the difference is itself informative.

---

## 4. Models under test

| | LLaVA-1.5-7B | LLaVA-1.5-13B | LLaDA-V-8B |
|---|---|---|---|
| HF id | `liuhaotian/llava-v1.5-7b` | `liuhaotian/llava-v1.5-13b` | `GSAI-ML/LLaDA-V` |
| Class | V-LLM (autoregressive) | V-LLM (autoregressive) | **V-DLM (masked diffusion)** |
| LLM backbone | Vicuna-1.5-7B | Vicuna-1.5-13B | `GSAI-ML/LLaDA-8B-Instruct` |
| `d_llm` (hidden) | 4096 | 5120 | 4096 |
| Vision tower `V_α` | `openai/clip-vit-large-patch14-336` | same | `google/siglip2-so400m-patch14-384` |
| `d_vis` | 1024 | 1024 | 1152 |
| `mm_vision_select_layer` | **−2** (penultimate) | −2 | verify in config |
| `mm_vision_select_feature` | **`patch`** (CLS is dropped) | `patch` | verify |
| Projector `F_β` | `mlp2x_gelu` (1024→d_llm→d_llm) | same | verify (LLaVA-NeXT-style likely) |
| `m` (image tokens) | **576** (24×24) | 576 | **verify at runtime** — SigLIP2 @384/p14 gives 729 base tokens, and if any-res/tiling is on, `m` is variable per image |
| Prompt template | `vicuna_v1` | `vicuna_v1` | check repo's conv template |

> **Two things to verify in code before anything else, and to assert on:**
> 1. `mm_vision_select_layer == -2` and `mm_vision_select_feature == 'patch'`. The −2 choice is why the "before projection" tap must be read from the **penultimate** ViT block, not the last one. Reading the last layer would compare against features LLaVA never sees.
> 2. LLaDA-V's actual `m`. If it varies per image (any-res tiling), pooling must be done per-example, and you cannot assume a fixed image-token span width.

---

## 5. Representation taps (the core of the harness)

For each image, one forward pass should yield **all** taps. Define them precisely:

| Tap | Name | Where | Shape (LLaVA-7B) | Poolings | Notes |
|---|---|---|---|---|---|
| **T0-c** | contrastive embedding | `V_α` last layer CLS → `visual_projection` → L2-norm | `[768]` | — (single vector) | This is the *actual CLIP/SigLIP2 contrastive space*. It is the reference ceiling and the only tap where zero-shot text matching is meaningful. **It is NOT the tap LLaVA consumes.** Keep them separate; conflating them is the most common error here. |
| **T0** | pre-projection vision features | `V_α` hidden_states[−2], patch tokens only (CLS dropped) | `[576, 1024]` | avg, CLS* | The "before projection" bar in FINER Fig. 5. *CLS is dropped by `select_feature='patch'`; extract it separately as an extra variant since it is the ViT's own pooled summary. |
| **T1** | post-projection, pre-LLM | `F_β(T0)` — i.e. `S_o` | `[576, 4096]` | avg, final | The "after projection" bar in FINER Fig. 5. **Caveat:** "final" here is just the bottom-right patch. The projector is position-wise, so no token has aggregated any other. Report it, but expect it to be weak and do not over-read it. |
| **T2** | post-LLM, image tokens only | last hidden layer, sliced to the image-token span → `H_o` | `[576, 4096]` | avg, final | "final" **is** meaningful here: causal attention means the last image token has attended over the whole image. |
| **T3** | post-LLM, all tokens | last hidden layer, full sequence | `[~620, 4096]` | avg, **final = the `:` token** | This is exactly Zhang et al.'s protocol and reproduces their 77.1% ImageNet number. The `final` variant is the state the answer is actually decoded from. |
| **T3-txt** | text-only control | `L_θ(E_φ(C_i))` with **no image**, prompt naming the class in text | `[n, 4096]` | avg, final | The modality-gap control. Measures separability of the same concepts presented textually. Gives the "textual modality" side of your hypothesis. |
| **T-mid** | layerwise sweep (ablation) | hidden_states at layers `{0, 8, 16, 24, 32}` (and `{0,10,20,30,40}` for 13B), image-token span | `[576, d]` | avg, final | Optional but strongly recommended — this is what turns "there is a drop" into "the drop is at layer k". **Costs 10 extra vectors/image, more than the whole core tap set — restrict to ImageNet-sub + CUB (§8.3B).** |

### 5.1 Getting the image-token span indices

This is the part that will bite you. In `haotian-liu/LLaVA`, `prepare_inputs_labels_for_multimodal()` splices projected image embeddings into `inputs_embeds` and **returns no index map**, so downstream you cannot tell which positions are image tokens.

Two acceptable fixes, in order of preference:

1. **Patch it to return the span.** Wrap `prepare_inputs_labels_for_multimodal` and have it also emit `(img_start, img_end)` per batch element. Cleanest, no assumptions.
2. **Reconstruct it.** With a single `<image>` placeholder at known token index `p` in the pre-expansion prompt and a fixed `m`, the span is `[p, p+m)`. Assert `seq_len == pre_len - 1 + m`. This breaks under any-res tiling — so guard it with the assert, don't trust it silently.

If you use the HF `LlavaForConditionalGeneration` port instead of the original repo, `input_ids` retains the expanded `image_token_index`, so the span is just `(input_ids == cfg.image_token_index).nonzero()`. **This is the easiest path — prefer the HF port if the numbers reconcile with the original repo on a 500-image spot check.**

### 5.2 Prompt (fixed, do not vary between taps)

Use Zhang et al.'s B.4 prompt so the numbers are comparable to theirs:

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>
What type of object is in this photo? ASSISTANT:
```

Note the trailing `:` with **no** trailing space — the `:` is the "last token" in the T3-final pooling. Verify with the tokenizer that it is in fact a standalone final token and log its id once.

---

## 6. Experiments

### 6.1 Experiment 1 — baseline generative accuracy

**Protocol (as decided): open-ended generation + text matching**, plus CLIP-style zero-shot for the encoders only.

- **Generation:** greedy (`do_sample=False`, `num_beams=1`, `max_new_tokens=32`), the prompt above. For LLaDA-V use its repo's default diffusion decoding settings; record `steps` and `gen_length` in the results row since they are not comparable to AR decoding.
- **Matching cascade**, applied in order, log which rung fired:
  1. normalize (lowercase, strip articles/punctuation, collapse whitespace, singularize)
  2. exact match against class name
  3. exact match against a hand-maintained **alias table** (`aliases.json`: e.g. `"boeing 737-800"` ↔ `"737-800"`, `"black_footed_albatross"` ↔ `"black footed albatross"`, ImageNet's multi-synonym labels split on `,`)
  4. substring containment (class name appears in the generation)
  5. nearest class by sentence embedding (`sentence-transformers/all-mpnet-base-v2`) with a cosine threshold `τ = 0.65`
  6. else → **unmatched**
- **Report three numbers per cell:** `strict` (rungs 1–3 only), `lenient` (rungs 1–5), and `%unmatched`. Never report a single number — the matcher is the single biggest source of irreproducibility in this literature.
- **Encoder baselines (T0-c):** standard CLIP zero-shot with the 80-prompt OpenAI ensemble for ImageNet and the `"a photo of a {c}."` + dataset-specific templates for FGVC sets. SigLIP2 uses sigmoid scoring but argmax over classes is identical in practice.

**Deliverable:** Table 1 — rows = {CLIP-L/14-336 ZS, SigLIP2-so400m ZS, LLaVA-7B, LLaVA-13B, LLaDA-V-8B}, cols = {ImageNet, CUB, Cars, Dogs, Aircraft}.

### 6.2 Experiment 2 — linear probes at every tap

**This is the main experiment.** For each (model, dataset, tap, pooling) train one linear classifier on frozen cached features.

Probe config (from Zhang et al. B.4):
```yaml
probe:
  head: linear            # nn.Linear(d, num_classes), no bias-free tricks, no BN
  standardize: true       # fit mean/std on train split only
  optimizer: adam
  lr: 1.0e-3
  batch_size: 512
  epochs: 500
  weight_decay: 0.0
  select: best_val_top1
  seeds: [0, 1, 2]        # report mean ± std; the spread matters for small FGVC sets
```
> A `sklearn.linear_model.LogisticRegression(max_iter=1000, C=…)` with L-BFGS converges much faster and lands within ~0.5% on these features. Use it for the sweep, and reproduce the headline cells with the torch config above so the numbers are directly comparable to the paper.

**Dimensionality fairness caveat — put this in the writeup.** Taps have different widths (1024 / 1152 / 4096 / 5120). A wider linear probe is strictly more expressive, so a *higher* score at a wider tap is not evidence of more information. Controls to run:
- report probe accuracy vs. a PCA projection of every tap to a common `d = 1024`;
- report accuracy at matched *train-set size* (probe capacity interacts with n);
- add a **k-NN probe** (k=20, cosine) alongside the linear probe. Linear probe measures *linear decodability*; kNN measures *metric structure*. If linear drops but kNN holds, information is present but linearly entangled — a different claim than information loss, and one that matters for your hypothesis.
- optionally a 1-hidden-layer MLP probe (2048 units) as an upper bound on decodability.

### 6.3 Experiment 3 — the modality-gap control (recommended, directly tests your thesis)

Everything above measures *class* separability. Your hypothesis is about **concept–attribute knowledge retrieval**. Add:

- **Text-side probe (T3-txt):** feed `"USER: What type of object is a {C_i}? ASSISTANT:"` with no image, probe the same taps. If image-side probes are far below text-side probes on the *same* label set, that is the modality gap, measured.
- **Attribute probe:** using FINER's attribute annotations (or CUB's 312 binary attributes, which are free and already aligned), train **multi-label** probes at each tap to predict attributes rather than class. Report mean AP. This is the most direct instrument for "concept–attribute knowledge retrieval signal" and it is what distinguishes your contribution from a re-run of Figure 5.
- **Gap metric:** report `Δ = acc(text tap) − acc(image tap)` per tap, and the **CKA / mutual-kNN alignment** between `H_o` and `H_c` for matched concepts.

---

## 7. Datasets

| Dataset | Granularity | Classes | Train | Test | Source |
|---|---|---|---|---|---|
| ImageNet-1k | coarse | 1000 | **subsample 100/class = 100k** | val 50k | HF `imagenet-1k` (gated) or local ILSVRC12 |
| CUB-200-2011 | fine | 200 | 5,994 | 5,794 | `caltech-ucsd birds`; also ships 312 attributes |
| Stanford Cars | fine | 196 | 8,144 | 8,041 | **Stanford host is dead** — use HF `tanganke/stanford_cars` or the Kaggle mirror |
| Stanford Dogs | fine | 120 | 12,000 | 8,580 | vision.stanford.edu/aditya86/ImageNetDogs |
| FGVC-Aircraft | fine | 100 (variant) | 6,667 trainval | 3,333 | `torchvision.datasets.FGVCAircraft` |
| *(optional, for exact FINER parity)* NABirds, iNaturalist | fine | 555 / varies | — | — | only if you want all six bars |

**Splits:** carve a validation split off train (10%, stratified, fixed seed 0) for probe model selection. Test splits are touched exactly once per final number.

**Preprocessing:** each model's own image processor, unmodified (`image_aspect_ratio: pad` for LLaVA-1.5). Do not normalize images yourself. Cache the processed pixel values only if disk is cheap; otherwise re-process.

---

## 8. Storage and runtime plan (1x A100/H100 80GB)

**The one rule that decides whether this is feasible: pool inside the extraction loop, before anything is written to disk.** Pooling collapses `[m, d]` to `[d]`, a 576x reduction for LLaVA. Everything below follows from that.

### 8.1 Image counts (train + test; train fits the probe, test produces the number)

| Dataset | Train | Test | Total |
|---|---|---|---|
| ImageNet-1k (100/class subsample + val) | 100,000 | 50,000 | 150,000 |
| CUB-200-2011 | 5,994 | 5,794 | 11,788 |
| Stanford Cars | 8,144 | 8,041 | 16,185 |
| Stanford Dogs | 12,000 | 8,580 | 20,580 |
| FGVC-Aircraft | 6,667 | 3,333 | 10,000 |
| **Total per model pass** | | | **208,553** |

### 8.2 Per-image cost

One image, LLaVA-7B, tap T2:

| | bytes |
|---|---|
| per-token `[576, 4096]` fp16 | **4.72 MB** |
| pooled `[4096]` fp16 | **8 KB** |

All eight tap/pooling combos pooled (`T0.avg`, `T0.cls` at `d_vis`; `T1/T2/T3` x `{avg, final}` at `d_llm`) = `2*1024 + 6*4096 = 26,624` floats = **52 KB/image**. Storing every stage of the network for one image costs less than a JPEG of it.

### 8.3 Totals

**A. Core (pooled, all models, all datasets, all taps):**

| Model | floats/img | KB/img | Total |
|---|---|---|---|
| LLaVA-1.5-7B | 26,624 | 52.0 | 10.34 GB |
| LLaVA-1.5-13B (d=5120) | 32,768 | 64.0 | 12.73 GB |
| LLaDA-V-8B (d_vis=1152) | 26,880 | 52.5 | 10.44 GB |
| CLIP-L + SigLIP2 (T0-c, T0.avg, T0.cls) | — | — | 2.44 GB |
| **Subtotal A** | | | **35.9 GB** |

**B. Layer sweep (T-mid), optional — but start cheap.**

*Why this exists (it is not in the original four-tap plan):* T0->T1 is one 2-layer MLP and T2->T3 is a slice of the same tensor, but **T1->T2 spans all 32 (or 40) LLM layers**. Three of the four intervals are tiny and one contains essentially the whole model. If the drop lands there, endpoint-only measurement yields "something in the LLM loses it" with no follow-up available. The sweep splits that interval into five and distinguishes *lossy projection the LLM never recovers from* (flat-and-low from layer 0) from *progressive discarding of visual detail as the stack converges on language-space semantics* (high early, decaying late). It also lets you compare the **depth profile of an AR model against a diffusion model**, which is the most VDLM-specific result available here and is invisible at the endpoints.

Cost is **disk only, not compute** — the hidden states already exist in the forward pass (`output_hidden_states=True`), and you pool them in the same loop.

**Default scope: CUB only.** 5 layers x 2 poolings = 10 extra vectors/image.

| Scope | LLaVA-7B | LLaVA-13B | LLaDA-V-8B | Total |
|---|---|---|---|---|
| **CUB only (11,788 imgs)** — do this first | 0.90 GB | 1.12 GB | 0.90 GB | **2.9 GB** |
| \+ ImageNet-sub (161,788 imgs) | 12.34 GB | 15.43 GB | 12.34 GB | 40.1 GB |
| All five datasets | — | — | — | ~52 GB |

Look at the CUB curve first; only spend the 40 GB on ImageNet if it shows something worth confirming at coarse granularity.

**C. Per-token cache for later re-pooling, optional.** Cache **T0 and T2 only, for 2,000 images** (1,000 ImageNet + 250 per FGVC set). Two reasons this is much cheaper than caching everything:
- **T1 is recoverable from T0 for free** — it is exactly `F_beta(T0)`, a 2-layer MLP, milliseconds to recompute.
- **T3's image span *is* T2**; only its ~44 text tokens are new, so store those separately.

| Model | MB/img | Total |
|---|---|---|
| LLaVA-7B | 5.97 | 11.66 GB |
| LLaVA-13B | 7.18 | 14.02 GB |
| LLaDA-V-8B | 6.11 | 11.93 GB |
| **Subtotal C** | | **37.6 GB** |

> For contrast, the naive version of this cache (7,000 images x all four taps per-token) is **332 GB across three models**. Do not do that.

| Scope | Disk |
|---|---|
| **Core only (A)** | **35.9 GB** |
| Core + layer sweep, CUB only (A+B) | **38.8 GB** |
| Core + layer sweep incl. ImageNet | 76.1 GB |
| Everything, full sweep (A+B+C) | 113.7 GB |

### 8.4 Format

One `.npy` (fp16) per `(model, dataset, split, tap, pooling)`, plus a shared `labels.npy` and a `manifest.json` recording model revision/commit, dtype, prompt hash, `m`, and span-derivation method. Layout:

```
features/
  llava-1.5-7b/
    cub/
      train/{T0.avg.npy, T0.cls.npy, T1.avg.npy, T1.final.npy, T2.avg.npy, T2.final.npy, T3.avg.npy, T3.final.npy}
      test/...
      labels.train.npy, labels.test.npy, manifest.json
```

### 8.5 Runtime

~208.5k images per model pass. At `m=576`, batch 32, bf16, expect roughly 40-60 img/s on an A100 for 7B -> **~60-85 min per model-pass**. 13B is ~2x (~2.5 h). LLaDA-V in `clean` mode is 7B-ish; in `sweep` mode ~5x on its restricted dataset pair. Budget **~1 day of GPU** for all extraction, plus a few hours for probes (each probe is GPU-minutes).

**Precision:** everything in **bf16**, `torch.no_grad()`, `model.eval()`. Do not mix fp16 and bf16 across taps - small hidden-state differences shift probe numbers by a few tenths and you will chase ghosts.

## 9. LLaDA-V specifics (V-DLM)

LLaDA-V is a **masked diffusion** LM, so "the hidden state" is a function of the denoising timestep and the mask pattern. There is no single canonical answer; the choice must be documented.

**Decision: default to the clean single forward pass, ship the mask variant behind a flag.**

- **`--llada-mode clean` (default).** Feed `[image tokens] + [prompt tokens]` with **no `[MASK]` tokens in the answer span**, one forward pass, take last-layer hidden states. Slice T2 at the image span, T3 over the full sequence with "final" = last prompt token. This is the closest structural analogue to the LLaVA probe: same inputs, same tap positions, one pass per image, so T0/T1/T2/T3 across the three models are comparable.
  *Trade-off:* it is **not** the state the model actually decodes from. LLaDA-V never runs this exact configuration at inference. If the probe is high here but generation is bad, you cannot immediately attribute that to decoding — the probed state is off-policy.

- **`--llada-mode masked --k 8`.** Append `k` `[MASK]` tokens as the answer span, take hidden states at the mask positions (mean over the k, and the first mask position separately). This is faithful to the decoding path and is the right tap if you want to make the H2/H3 argument for the VDLM.
  *Trade-off:* introduces `k` as a free parameter, and the LLaVA side has no analogue, so cross-model comparison at this tap needs a caveat.

- **`--llada-mode sweep`.** Extract at mask ratios `{0.0, 0.25, 0.5, 0.75, 1.0}` over the answer span and plot probe accuracy vs. mask ratio. ~5× extraction cost, and it is the most interesting VDLM-specific result in the study — a curve here is a paper figure in its own right. **Run this on CUB + ImageNet-subsample only**, not on everything.

**Bidirectional attention caveat:** LLaDA's attention is not causal. Therefore at T2 the "final image token" is **not** privileged the way it is in LLaVA — every image token sees every other. Expect `T2.avg ≈ T2.final` for LLaDA-V and a gap for LLaVA. If you see the LLaVA-like pattern in LLaDA-V, you have a bug in the span slicing. Note this in the writeup; it is a free correctness check.

---

## 10. Repo layout and interfaces

```
vlm-probe/
  configs/
    models/{llava15_7b,llava15_13b,lladav_8b,clip_l14_336,siglip2_so400m}.yaml
    datasets/{imagenet,cub,cars,dogs,aircraft}.yaml
    probe.yaml
  src/
    models/
      base.py            # TapExtractor ABC
      llava.py           # LLaVATapExtractor
      llada_v.py         # LLaDAVTapExtractor  (handles --llada-mode)
      encoders.py        # CLIP / SigLIP2 (T0-c, T0)
    data/
      registry.py        # get_dataset(name, split) -> (PIL, label_idx); classnames; aliases
      templates.py       # zero-shot prompt ensembles
    extract.py           # CLI: writes features/ tree
    probe.py             # CLI: reads features/, writes results/*.json
    evaluate_generative.py  # CLI: Experiment 1
    match.py             # the matching cascade + alias table
    figures.py           # Figure 5 recreation + layer sweep plot
  tests/
    test_span.py         # asserts image-token span correctness on 20 images/model
    test_shapes.py
  results/
  features/
```

**The one interface that matters** — write this first, everything else hangs off it:

```python
# src/models/base.py
@dataclass
class Taps:
    t0c:  Optional[Tensor]   # [B, d_contrastive]      contrastive embedding
    t0:   Tensor             # [B, m, d_vis]           pre-projection, penultimate ViT layer, patch tokens
    t0_cls: Optional[Tensor] # [B, d_vis]              ViT CLS (dropped by LLaVA, kept for reference)
    t1:   Tensor             # [B, m, d_llm]           post-projector  (== S_o)
    t2:   Tensor             # [B, m, d_llm]           last LLM layer, image span (== H_o)
    t3:   Tensor             # [B, T, d_llm]           last LLM layer, full sequence
    img_span: Tuple[int,int] # (start, end) into t3
    mid:  Dict[int, Tensor]  # layer_idx -> [B, m, d_llm]   optional layer sweep

class TapExtractor(ABC):
    @abstractmethod
    def forward_taps(self, images: List[PIL.Image], prompt: str,
                     layers: Sequence[int] = ()) -> Taps: ...
```

Pooling happens in `extract.py`, immediately, before anything is written to disk.

CLI shape:
```bash
python -m src.extract  --model llava15_7b --dataset cub --split train --taps T0,T1,T2,T3 --pool avg,final
python -m src.probe    --model llava15_7b --dataset cub --tap T2 --pool avg --head linear --seeds 0,1,2
python -m src.evaluate_generative --model llava15_7b --dataset cub
python -m src.figures  --figure fig5 --datasets cub,cars,dogs,aircraft,imagenet
```

---

## 11. Sanity checks — run these before trusting any number

1. **Reproduce Zhang et al.:** LLaVA-1.5-7B, ImageNet, tap **T3**, pooling **final** (the `:` token), linear probe → expect **≈77%** top-1. CLIP-L T0-c linear probe → expect **≈85%**. If you are more than ~3 points off, the bug is in the prompt, the span, or the layer selection — in that order of likelihood.
2. **Reproduce the token-position collapse:** probe T3 at position 0 → should be near chance (Zhang et al. report 0.7% on Flowers102). If position 0 probes well, you are slicing the wrong tensor.
3. **Reproduce FINER's direction:** T0.avg > T1.avg on all four FGVC sets. Magnitude may differ (they use different probe settings) but the sign must hold.
4. **T1 identity check:** `F_β(T0)` computed standalone must equal T1 read out of the live forward pass to within bf16 tolerance. Catches select_layer/select_feature mismatches.
5. **Span check (`tests/test_span.py`):** for 20 images, assert `img_span` width == `m`, and assert that zeroing the pixel values changes `t3[:, img_span]` and does **not** change `t3[:, :img_span[0]]`.
6. **LLaDA bidirectionality check:** `T2.avg ≈ T2.final` for LLaDA-V, but not for LLaVA (see §9).
7. **Label leakage:** confirm the prompt never contains the class name in Experiment 2, and that ImageNet subsampling is drawn from **train**, never val.

---

## 12. Deliverables

- `results/table1_generative.md` — Experiment 1, with strict/lenient/%unmatched.
- `results/table2_probes.csv` — long format: `model, dataset, tap, pooling, head, seed, split, top1, top5`.
- `figures/fig5_recreation.pdf` — grouped bars, x = dataset, series = {T0.avg, T1.avg, T2.avg, T2.final, T3.final}, y = probe top-1, with the CLIP T0-c zero-shot and the generative accuracy drawn as horizontal reference marks per dataset.
- `figures/layer_sweep.pdf` — x = LLM layer index, y = probe top-1, one line per (model, dataset). This is the figure that localizes the loss.
- `figures/modality_gap.pdf` — image-tap vs text-tap probe accuracy, per tap.
- `WRITEUP.md` — 2 pages: which of H1/H2/H3 the curve supports, per model class, with the dimensionality caveat stated explicitly.

---

## 13. Known traps

- **Penultimate vs last ViT layer.** LLaVA reads layer −2. Probing layer −1 and calling it "before projection" invalidates the T0→T1 comparison.
- **CLS token.** LLaVA drops it (`select_feature='patch'`). The CLIP *contrastive* embedding is derived from CLS through `visual_projection`. So the strong "CLIP zero-shot" number and the "before projection" probe number come from **different tensors**. Keep T0-c and T0 in separate columns and say so in the caption; conflating them is how you accidentally report a much larger projector drop than exists.
- **Any-res / tiling.** LLaVA-1.5 does not tile (fixed 576), but LLaDA-V may. Never hardcode `m` for LLaDA-V.
- **`sklearn` vs torch probe.** They differ by ~0.5%. Pick one per table; do not mix within a figure.
- **Stanford Cars download.** The canonical URL is dead. Pin a mirror in the dataset config and record its checksum.
- **Padding in batched extraction.** Left- vs right-padding shifts the image span and breaks "last token" pooling. Either extract with batch size 1 (slow but safe) or store per-example spans and mask correctly. Add an assert.
- **Probe on unstandardized features.** Post-LLM hidden states have wildly different per-dimension scales (RMSNorm outputs, plus outlier dims). Always standardize; without it the 500-epoch Adam config will underfit badly and you will report a fake drop.

---

## 14. Open decisions (flagged, not blocking)

- Whether to add NABirds + iNaturalist for exact six-bar parity with FINER Fig. 5. Costs ~1 extra day of extraction; buys direct visual comparability.
- Whether the attribute probe (§6.3) uses CUB's native 312 attributes or FINER's released attribute set. CUB's is free and immediate; FINER's is the better match to the framing.
- Whether to include a **trained-projector control**: retrain only `F_β` with a classification objective and re-probe, to test whether the projector's capacity or its training objective is the binding constraint. This is the natural follow-up experiment if H1 is supported.

---

## Sources

- [FINER: Investigating and Enhancing Fine-Grained Visual Concept Recognition in LVLMs (Kim & Ji, arXiv:2402.16315)](https://arxiv.org/abs/2402.16315)
- [Why are Visually-Grounded Language Models Bad at Image Classification? (Zhang et al., NeurIPS 2024, arXiv:2405.18415)](https://arxiv.org/abs/2405.18415)
- [Finedefics (arXiv:2501.15140)](https://arxiv.org/abs/2501.15140)
- [LLaVA Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)
- [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V)
