---
layout: post
title:  "WordFor: A reverse dictionary that runs entirely in your browser"
description: "Building a free, private reverse dictionary with sentence embeddings, ONNX Runtime, and static model inference -- zero server-side compute."
image: images/wordfor.webp
invert_thumbnail_dark: true
date:   2026-04-12 15:00:00 -0700
categories: deep-learning nlp transformers embeddings side-project
author: Zeeshan Khan Suri
published: true
comments: true
---

You know the feeling. You *know* the word exists, you can describe it perfectly, but your brain refuses to recall it. So you Google *"word for fear of being forgotten"*, scroll through listicles, and maybe five minutes later find **athazagoraphobia**.

I got tired of that loop. So I built [**WordFor**](https://wordfor.xyz){:target="_blank"}: a reverse dictionary where you describe a concept and get the word. No ads, no sign-ups, no data leaving your device. The AI model runs entirely in the browser.

What started as a weekend embedding search turned into a data engineering project. The model was the easy part, the hard part was building a dictionary worth searching.

## The approach: semantic search over a dictionary

A dictionary maps words to definitions. I want the reverse: map *descriptions* to *words*. Given *"a feeling of longing for the past"*, the system should return **nostalgia**. Keyword matching fails here -- the query barely overlaps with the definition (*"a bittersweet longing for things, persons, or situations of the past"*). I need *semantic* similarity.

Modern sentence embedding models (think of it as a converter of discrete  words -> continuous vectors) are perfectly suited for this. I encode every dictionary definition into a dense vector at build time. At query time, I encode the user's description with the same model and rank definitions by cosine similarity. The challenge is doing all of this in the browser, at interactive speed, over 350,000 definitions.

## The data

With data-centric AI principles, the quality of the dictionary determines the quality of results. No amount of model tuning can fix bad definitions. I needed high-quality, redistributable dictionary data. The following sources aligned with the license requirements:

- [Open English WordNet 2025](https://en-word.net/){:target="_blank"} (120k+ synsets, CC-BY 4.0) -- the primary source with modern, well-structured definitions
- [Webster's 1913 Unabridged](https://github.com/adambom/dictionary){:target="_blank"} (public domain, ~100k entries) -- comprehensive coverage of English vocabulary
- [GCIDE](https://gcide.gnu.org.ua/){:target="_blank"} Webster 1913 portion (public domain, ~73k entries) -- supplementary Webster entries not in the JSON export
- [Century Dictionary](https://en.wikipedia.org/wiki/Century_Dictionary){:target="_blank"} (1889--1911, public domain, ~161k entries) -- the largest single source, strong on technical and literary vocabulary

For enrichment (not redistributed), I also use:
- [Moby Thesaurus II](https://en.wikipedia.org/wiki/Moby_Project){:target="_blank"} (public domain, 30k roots, 2M+ synonyms) -- synonym data stored alongside definitions
- [Moby Part-of-Speech II](https://github.com/elitejake/Moby-Project){:target="_blank"} (public domain, 233k words) -- build-time coverage signal
- [Moby Words II COMMON.TXT](https://github.com/elitejake/Moby-Project){:target="_blank"} (74,550 words appearing in 2+ major dictionaries) -- build-time quality signal
- [Wiktionary](https://kaikki.org/){:target="_blank"} (CC-BY-SA 3.0, 1.3M headwords) -- build-time quality scoring and embedding enrichment
- [ConceptNet 5.7](https://conceptnet.io/){:target="_blank"} (CC-BY-SA 4.0, 878k English concepts) -- knowledge graph signals for quality scoring

Building a dictionary from five sources written across three centuries is less "merge and ship" and more "archaeological dig." Each source has its own quirks, and getting clean data out required source-specific parsers and a multi-stage cleaning pipeline.

### Source-specific parsing

**OEWN** is the cleanest source -- modern JSON with structured synsets, part-of-speech tags, and cross-references. Minimal cleaning needed.

**Webster's 1913** comes as a monolithic JSON file with raw 19th-century formatting. Definitions contain archaic abbreviations, inline etymology in parentheses, and cross-references like *"See Def. 2"* that make no sense in isolation. I strip HTML tags, normalize whitespace, and filter out the stub definitions that just point elsewhere.

**GCIDE** is trickier. It's the GNU Collaborative International Dictionary of English -- a community-maintained extension of Webster's 1913. The codebase contains both the original public-domain Webster entries and GPL-licensed additions from WordNet and other contributors. I wrote `parse_gcide.py` to extract *only* the public-domain portion, checking source attribution tags to avoid GPL contamination.

**Century Dictionary** was the most challenging. The source is a Markdown conversion of OCR'd scans from 1889--1911. Issues included:
- Contributor attribution text embedded mid-definition (delimited by `»` and `«` markers)
- ALL-CAPS section headings leaking into definition text
- OCR garbage: strings like `"DICTIONARY (>F THE ENGLISH LAXGDACxE 1-2187"` appearing as definitions
- Foreign-language text fragments from adjacent dictionary pages

I built a parser with regex-based stripping of attribution markers and section headings, plus a safety net in the loader that catches remaining garbage patterns. Even after automated cleaning, I found entries that had to be patched manually. For e.g. the definition for "standard" was an OCR artifact of a page header, and "foreign" had been replaced by a biographical passage about a Finnish artist.

### The merge pipeline

With four sources parsed, the build pipeline:

1. **Loads** all sources and normalizes each entry to a common schema: `{w: [word, ...variants], d: definition, p: part-of-speech, s: [synonyms]}`
2. **Enriches** entries with Moby Thesaurus synonyms (matched by headword)
3. **Deduplicates** across sources -- when the same word appears in multiple dictionaries, I keep each unique definition sense as a separate entry (per-sense format). This means "cut" appears 53 times, once for each distinct meaning
4. **Groups morphological variants** using a suffix-stripping stemmer (retrospective/retrospection, comprehension/comprehensible) with a definition-overlap guard to prevent false merges
5. **Computes quality scores** from cross-source agreement, definition richness, and knowledge graph signals
6. **Injects LLM-augmented definitions** for words absent from all public-domain sources (see below)
7. **Encodes embeddings** with the teacher model

After deduplication and merging, the dictionary contains over **350,000** definition entries.

### Quality scoring

Not all 350k definitions are equally useful. Compound words like "art teacher" shouldn't outscore "teacher," and ultra-generic words like "time" or "place" shouldn't dominate specific queries. Rather than engineering the ranking function, I improve the *data*: a quality weight $q$ for every definition, computed at build time from multiple signals:

**Cross-source agreement.** Each word is checked against up to eight sources (OEWN, Webster's, GCIDE, Century, Wiktionary, ConceptNet, Moby POS, Moby COMMON). Words confirmed by more sources get a small boost.

**ConceptNet knowledge graph.** ConceptNet's 3M English edges provide two signals: *centrality* (well-connected concepts score higher) and *relation diversity* (concepts participating in many relation types -- IsA, HasProperty, Synonym -- are better-defined).

**Definition quality.** Heuristics penalize short (<8 tokens), overly long (>25 tokens), and generic definitions ("a kind of ..."). An IDF-based richness score rewards definitions using specific vocabulary.

**Compound penalty.** Multi-word entries where all components exist as standalone words get penalized.

**Hub word penalty.** Words that appear in more than 5% of all definitions (like "time," "place," "person") get a 0.80--0.90x multiplier. This is applied *after* the $q^{0.1}$ dampening so it has real effect rather than being compressed. A guard clause exempts words with highly specific definitions (high IDF score) -- so "alarm" (specific: *"fear from a sense of danger"*) keeps its score even though "alarm" appears frequently in other definitions.

The raw score is dampened via $q^{0.1}$ to act as a soft re-ranking signal rather than a hard filter:

{:refdef: style="text-align: center;"}
$$\text{score}_{\text{final}} = \text{cosine}(q_{\text{user}}, e_{\text{def}}) \cdot q^{0.1}$$
{: refdef}

On the 67-query test set, quality weighting alone improved the base potion model by +7.8% MRR: matching a fine-tuned model without changing any weights.

## The embedding model

[MTEB](https://huggingface.co/spaces/mteb/leaderboard) is the Massive Text Embedding Benchmark which ranks embedding models. The top performers are huge; I need one that can run in a browser.

I evaluated four small transformer models in symmetric mode (same model for both queries and definitions):

| Model | Params | MRR | Hit@6 |
|-------|:---:|:---:|:---:|
| [mdbr-leaf-mt](https://huggingface.co/MongoDB/mdbr-leaf-mt){:target="_blank"} | 23M | 0.640 | 29/40 |
| [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs){:target="_blank"} | 22.6M | 0.630 | 29/40 |
| [bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2){:target="_blank"} | 17.4M | 0.549 | 27/40 |
| [paraphrase-albert-small-v2](https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2){:target="_blank"} | 11.7M | 0.539 | 26/40 |
| [paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2){:target="_blank"} | 17.4M | 0.516 | 25/40 |

*(Early evaluation on 40 pilot queries before the full 67-query test set was built.)*

I went with [mdbr-leaf-mt](https://huggingface.co/MongoDB/mdbr-leaf-mt){:target="_blank"}, which also has the following desirable properties:


### Asymmetric retrieval

The [mdbr-leaf-mt](https://huggingface.co/MongoDB/mdbr-leaf-mt){:target="_blank"} model is distilled from its big brother [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1){:target="_blank"}, and their embeddings align. I exploit this for **asymmetric retrieval**[^1]:

- **Build time**: [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1){:target="_blank"} (335M parameters) encodes definitions into 1024-dimensional embeddings.
- **Runtime**: [mdbr-leaf-mt](https://huggingface.co/MongoDB/mdbr-leaf-mt){:target="_blank"} (22M parameters) encodes queries into the same embedding space. It runs via [Transformers.js](https://huggingface.co/docs/transformers.js){:target="_blank"} + ONNX Runtime WASM in the browser.

Definitions are encoded *once* at build time -- I can afford a large model. Only the query model runs in the browser, so it must be small. The teacher-student pairing is what makes this work.

### Matryoshka truncation

Serving 350k $\times$ 1024 float32 vectors (488 MB) to a browser is impractical. mxbai-embed-large supports **Matryoshka Representation Learning (MRL)**[^2] -- leading dimensions carry the most information, so I truncate and re-normalize:

{:refdef: style="text-align: center;"}
$$\mathbf{v}_{d} = \frac{\mathbf{v}_{1:d}}{\|\mathbf{v}_{1:d}\|}$$
{: refdef}

Evaluated on the 67-query test set:

| Dimensions | MRR | Hit@1 | Hit@6 |
|:---:|:---:|:---:|:---:|
| 1024 | 0.673 | 24/40 | 32/40 |
| 768 | 0.672 | 24/40 | 33/40 |
| 512 | 0.665 | 23/40 | 33/40 |
| **384** | **0.653** | **23/40** | **32/40** |
| 256 | 0.659 | 23/40 | 32/40 |
| 128 | 0.594 | 20/40 | 30/40 |

*(MRL evaluation was performed early on with the 40-query pilot set; final deployment is validated on the full 67-query set.)*

256d actually edges out 384d on MRR (0.659 vs 0.653) -- the Matryoshka ordering isn't strictly monotonic on a small test set. I went with 384d for the slightly better Hit@6 and to stay consistent with the query model's native output.

## Quantization

At 384d, float32 embeddings are still 176 MB. Per-dimension scalar quantization to int8:

{:refdef: style="text-align: center;"}
$$q_d = \mathrm{round}\!\left(\frac{v_d - \min_d}{\max_d - \min_d} \times 255\right) \in [0, 255]$$
{: refdef}

At query time, the dot product decomposes into a scaled int8 dot product plus a per-query constant -- cheap to compute. Int4 quantization halves the file size further while matching or beating int8 quality -- likely because the coarser quantization grid acts as light regularization, smoothing over noise in the embedding space.

## 1-bit binary quantization

Int8 is 4x smaller than float32, but I can go further. **Binary quantization** maps each dimension to a single bit (positive -> 1, negative -> 0), replacing cosine similarity with **Hamming distance** via XOR + popcount -- bitwise operations that modern CPUs execute in a single instruction.

The naive approach (threshold at zero) loses information because the dimension means aren't centered. **Iterative Quantization (ITQ)**[^4] learns an orthogonal rotation $R$ that minimizes quantization error:

{:refdef: style="text-align: center;"}
$$\mathbf{b} = \text{sign}\!\left((\mathbf{v} - \boldsymbol{\mu}) \cdot R\right)$$
{: refdef}

The rotation matrix is learned offline in ~50 iterations on a subsample of the definition embeddings. At query time, the query vector gets the same rotation before binarization.

**Does 384d matter for binary?** MRL concentrates semantic information in leading dimensions, but binary quantization is lossy -- more bits might help. I tested pure binary retrieval at different MRL truncations on a 3K-entry subsample (62 valid queries):

| Dimensions | Pure Binary MRR | Hit@1 |
|:---:|:---:|:---:|
| 1024 | 0.802 | 50/62 |
| 768 | 0.809 | 50/62 |
| **384** | **0.821** | **51/62** |
| 192 | 0.728 | 42/62 |

384d is optimal -- more dimensions past this point add noise rather than signal under binary quantization, consistent with MRL concentrating information in leading dimensions.

**Two-stage binary + reranking.** Pure binary retrieval at 384d is fast (~13ms for 350k entries) but loses some precision. A two-stage pipeline recovers it: binary Hamming distance narrows from 350k to the top 500 candidates, then a dense dot product reranks them. I tested int4, int3, and int8 reranking -- int3 (3-bit quantization, 8 dimensions packed per 3 bytes) achieves the best MRR (0.644), beating both int4 (0.609) and int8.

**Desktop** uses binary + int3 rerank (~8 MB binary + 75 MB int3). **Mobile** uses pure binary scoring only (~8 MB), keeping the total download under 30 MB.

Final deployed results on the 67-query test set:

| Config | MRR | Hit@1 | Hit@6 |
|--------|:---:|:---:|:---:|
| **binary + int3 rerank (desktop)** | **0.644** | **37/67** | **52/67** |
| pure binary ITQ (mobile) | 0.577 | -- | -- |

## The mobile problem

On iOS Safari, the 22M-parameter ONNX model fails to load -- both q8 and q4f16 quantizations produce `RangeError: Out of Memory`. WASM on iOS has strict memory limits, and the model's runtime footprint exceeds them. WebGPU isn't available on iOS either.

## Lite mode: static embeddings in pure JavaScript

For devices where the ONNX model can't load, WordFor falls back to a static embedding model -- a [Model2Vec](https://github.com/MinishLab/model2vec){:target="_blank"} architecture distilled from the teacher model. Inference is pure JavaScript:

1. WordPiece tokenize the query
2. Look up each token's 256d vector from a float16 matrix (~15 MB)
3. Mean pool and L2-normalize

Sub-1ms, zero dependencies.

## Knowledge distillation for lite mode

Off-the-shelf static models like [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M){:target="_blank"} are trained on general text. For a reverse dictionary, I can do better by distilling directly from the teacher model.

**Step 1: Distill the teacher.** Model2Vec's [`distill()`](https://github.com/MinishLab/model2vec){:target="_blank"} runs a forward pass through mxbai-embed-large-v1 for each vocabulary token and applies PCA to produce a 256d static embedding matrix. This gives the lite model a starting point that captures the teacher's embedding space -- the same space the full-mode definition embeddings live in.

**Step 2: Fine-tune on dictionary data.** sentence-transformers' `StaticEmbedding` module wraps the embedding matrix as a differentiable `torch.nn.EmbeddingBag`[^3]. Training pipeline:

1. **Data**: [Open English WordNet 2025](https://en-word.net/){:target="_blank"} (CC-BY 4.0, 115k entries) + [Webster's 1913](https://github.com/adambom/dictionary){:target="_blank"} (public domain, 70k entries) + [Wiktionary](https://kaikki.org/){:target="_blank"} (CC-BY-SA 3.0, 308k entries filtered to words already in the main dictionary). Total: 493k entries, 2.3M training pairs.
2. **Pairs**: Each (word, definition) becomes a positive pair -- the word is the "query" and the definition is the "passage."
3. **Loss**: `MultipleNegativesRankingLoss` with in-batch negatives -- for a batch of 2048, each pair has 2047 negatives for free.
4. **Decontamination**: Cross-deduplication removes training definitions too similar (cosine > 0.85) to the 67 evaluation queries.
5. **No deduplication**: Near-duplicate definitions from different sources are kept as natural data augmentation.

The embedding matrix is the only trainable parameter -- 30k tokens $\times$ 256 dimensions. No attention layers, no positional encodings, just a lookup table getting nudged by contrastive gradients. Training takes ~27 minutes on a single GPU with early stopping.

| Version | Base model | Training data | MRR | Hit@1 | Hit@6 |
|---------|-----------|--------------|:---:|:---:|:---:|
| potion-base-8M (off-the-shelf) | potion-base-8M | -- | 0.462 | 25/67 | 38/67 |
| potion-base-8M fine-tuned | potion-base-8M | OEWN + Webster's + Wikt | 0.513 | 29/67 | 41/67 |
| **distilled-mxbai fine-tuned (deployed)** | **distilled from mxbai-embed-large** | **OEWN + Webster's + Wikt** | **0.482** | **26/67** | **40/67** |

*(MRR reflects the current 350k-entry dictionary. After dictionary expansion from ~120k to 350k entries, more definitions compete per query. The distilled model compensates with better vocabulary coverage from the teacher.)*

Knowledge distillation from the teacher gives +3.3% MRR at baseline over potion-base-8M. Fine-tuning amplifies the benefit. The runtime is unchanged -- same WordPiece tokenization, same mean-pooling, same sub-millisecond inference. Only the embedding matrix weights are different.

## Rust WASM for lite mode inference

The pure JavaScript Model2Vec implementation works, but it has a weakness: a hand-rolled WordPiece tokenizer that doesn't match the original HuggingFace tokenizer exactly. Edge cases in normalization, unknown token handling, and subword splitting can produce slightly different embeddings.

[model2vec-rs](https://github.com/MinishLab/model2vec-rs){:target="_blank"} is an official Rust implementation of Model2Vec that includes the full HuggingFace tokenizer via the [tokenizers](https://github.com/huggingface/tokenizers){:target="_blank"} crate. Compiling to WebAssembly gives exact tokenizer parity with the Python training pipeline, in a 1.9 MB `.wasm` binary.

The WASM wrapper exposes a simple interface:

```rust
#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(tok: &[u8], model: &[u8], config: &[u8]) -> Self { ... }

    pub fn encode_single(&self, text: &str) -> Vec<f32> { ... }
}
```

At runtime, WordFor tries the WASM model first and falls back to pure JS if it fails. The WASM path is faster (~0.1ms vs ~0.3ms per query) and produces embeddings that exactly match Python -- verified against reference vectors with max difference < 0.001.

## Results

The full evaluation on a 67-query test set covering common words, rare phobias, scientific terms, abstract concepts, and near-miss pairs (active/passive semantics, antonyms, domain specificity):

**Full mode** (asymmetric: mdbr-leaf-mt queries, mxbai-embed-large definitions, 350k entries):

| Config | MRR | Hit@1 | Hit@6 |
|--------|:---:|:---:|:---:|
| **binary + int3 rerank (deployed desktop)** | **0.644** | **37/67** | **52/67** |
| pure binary ITQ (mobile) | 0.563 | 30/67 | 52/67 |

**Lite mode** (distilled-mxbai fine-tuned, 256d, int4 scoring):

| Config | MRR | Hit@1 | Hit@6 |
|--------|:---:|:---:|:---:|
| **distilled-mxbai fine-tuned (deployed)** | **0.566** | **33/67** | **42/67** |
| distilled-mxbai base (no fine-tuning) | 0.503 | 27/67 | 42/67 |

Note: full-mode and lite-mode evaluations use different embedding sets (asymmetric vs symmetric), so MRR values are not directly comparable across blocks.

### Browser benchmarks

Scoring latency over 350k entries (average of 5 queries):

| | Lite (Potion int4) | Full binary + int3 rerank | Full pure binary | Full int8 only |
|---|:---:|:---:|:---:|:---:|
| **Latency** | ~46 ms | ~46 ms | ~13 ms | ~76 ms |
| **Data size** | ~45 MB | ~43 MB | ~8 MB | ~68 MB |

Full mode also requires the mdbr-leaf-mt ONNX model (~22 MB, q8 WASM, ~50ms/query inference). Mobile uses pure binary to keep total download under 30 MB.

Deployed configuration: binary + int3 rerank on desktop, pure binary on mobile, automatic fallback to lite if the ONNX model fails to load.

## Architecture

```
    BUILD TIME (GPU)                    BROWSER (runtime)
    ----------------                    -----------------
    mxbai-embed-large-v1 (335M)        Full: mdbr-leaf-mt (22M) ONNX WASM
    -> 350k defs -> 1024d -> MRL 384d   -> query -> 384d
    -> ITQ binary (8 MB)                -> XOR+popcount 350k -> top 500
    -> int3 (75 MB, desktop only)       -> int3 rerank -> ~46ms (desktop)
                                        -> pure binary only -> ~13ms (mobile)

    distilled-mxbai (Model2Vec distill) Lite: model2vec-rs WASM (or pure JS)
    -> fine-tuned on all sources          -> query -> 256d -> dot 350k x q
    -> 350k defs -> 256d -> int4          -> ~46ms
    -> 23 MB

    Quality scoring (8 sources):
    OEWN + Webster + GCIDE + Century
    + Wiktionary + ConceptNet + Moby POS + Moby COMMON
    -> per-entry q weight
```

## What makes it fast

- **1-bit binary scoring**: XOR + popcount over 350k entries in ~13ms -- 6x faster than int8 dot products
- **Two-stage retrieval**: binary first-pass narrows to 500 candidates, int3 rerank recovers full quality
- **Int3/Int4 quantization**: 5-8x smaller files than float32 -- and *better* MRR than int8
- **Selective loading**: mobile skips the int3 rerank file entirely; full mode skips potion data
- **Service Worker caching**: cache-first after first visit
- **Model warmup**: dummy inference during loading avoids first-query compilation stall

## Privacy

Everything runs client-side. Static files served from GitHub Pages through Cloudflare CDN (see my post on [making GitHub Pages GDPR-compliant with Cloudflare](/github-pages-cloudflare-gdpr)). No server, no database, no cookies. Analytics via [GoatCounter](https://www.goatcounter.com/){:target="_blank"} -- cookie-free, no personal data.

## What's next

- **More data sources**: Chambers's Twentieth Century Dictionary (1901, public domain via Gutenberg) and frequency-weighted quality signals from Wikipedia Word Frequency data could further improve coverage and ranking.
- **Smarter deduplication**: The current suffix-stripping stemmer handles most morphological variants (retrospective/retrospection, incomprehension/uncomprehensive), but more sophisticated approaches -- full lemmatization, etymological grouping -- could reduce noise further.
- **Expanded LLM augmentation**: 61 definitions cover the most glaring gaps, but thousands of terse entries (<25 characters) remain candidates for improved definitions.

## Try it

**[wordfor.xyz](https://wordfor.xyz){:target="_blank"}** -- describe a concept, find the word.

If you find it useful, consider a share or a [Ko-fi](https://ko-fi.com/zshn25){:target="_blank"}.

___

&copy; Zeeshan Khan Suri, [<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-nd.eu.svg" width="60"/>  CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

If this article was helpful to you, consider citing

```bibtex
@misc{suri_wordfor_2026,
      title={WordFor: A reverse dictionary that runs entirely in your browser},
      url={https://zshn25.github.io/wordfor-reverse-dictionary},
      journal={Curiosity},
      author={Suri, Zeeshan Khan},
      year={2026},
      month={Apr}}
```

# References

[^1]: Asymmetric retrieval using the mxbai-embed-large + mdbr-leaf-mt pairing from [MongoDB's retrieval model suite](https://huggingface.co/MongoDB/mdbr-leaf-mt){:target="_blank"}.

[^2]: Kusupati et al., [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147){:target="_blank"}, NeurIPS 2022.

[^3]: Sentence-Transformers' [`StaticEmbedding`](https://sbert.net/docs/package_reference/sentence_transformer/models.html#staticembedding){:target="_blank"} module. See also: Minish Lab, [Model2Vec: Distill a Small Fast Model from any Sentence Transformer](https://huggingface.co/blog/Pringled/model2vec){:target="_blank"}.

[^4]: Gong et al., [Iterative Quantization: A Procrustean Approach to Learning Binary Codes](https://ieeexplore.ieee.org/document/6296665){:target="_blank"}, TPAMI 2013.

*[MRL]: Matryoshka Representation Learning
*[ONNX]: Open Neural Network Exchange
*[WASM]: WebAssembly
*[int8]: 8-bit integer quantization
*[MRR]: Mean Reciprocal Rank
*[OOM]: Out of Memory
*[ITQ]: Iterative Quantization
*[MTEB]: Massive Text Embedding Benchmark
