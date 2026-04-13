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

You know the feeling. You're writing and you *know* the word exists -- you can describe it perfectly, but your brain won't retrieve it. So you Google *"word for fear of being forgotten"*, scroll through listicles, and maybe five minutes later find **athazagoraphobia**.

I got tired of that loop. So I built [**WordFor**](https://wordfor.xyz){:target="_blank"} -- a reverse dictionary where you describe a concept and get the word. No ads, no sign-ups, no data leaving your device. The AI model runs entirely in the browser.

## Semantic search over a dictionary

A reverse dictionary maps *descriptions* to *words*. Given *"a feeling of longing for the past"*, it should return **nostalgia**. Keyword matching fails here -- the query barely overlaps with the definition (*"a bittersweet longing for things, persons, or situations of the past"*). We need *semantic* similarity.

The approach: encode every dictionary definition into a dense vector (embedding) at build time. At query time, encode the user's description and rank definitions by cosine similarity. The dictionary combines [Open English WordNet 2025](https://en-word.net/){:target="_blank"} (120k+ synsets, CC-BY 4.0) and [Webster's 1913 Unabridged](https://github.com/adambom/dictionary){:target="_blank"} (public domain) -- over 176,000 definitions after deduplication, enriched with [Moby Thesaurus](https://en.wikipedia.org/wiki/Moby_Project){:target="_blank"} synonyms. The challenge is doing all of this in the browser, at interactive speed.

## Asymmetric retrieval

Definitions are encoded *once* at build time -- we can afford a large model. Only the query model runs in the browser, so it must be small. This naturally leads to **asymmetric retrieval**[^1]:

- **Build time**: [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1){:target="_blank"} (335M parameters) encodes definitions into 1024-dimensional embeddings.
- **Runtime**: [mdbr-leaf-mt](https://huggingface.co/MongoDB/mdbr-leaf-mt){:target="_blank"} (22M parameters) encodes queries into the same embedding space. It runs via [Transformers.js](https://huggingface.co/docs/transformers.js){:target="_blank"} + ONNX Runtime WASM in the browser.

The key is that mdbr-leaf-mt was distilled to map queries into mxbai-embed-large's embedding space -- the teacher-student pairing is what makes asymmetric retrieval work.

## Matryoshka truncation

Serving 120k $\times$ 1024 float32 vectors (488 MB) to a browser is impractical. mxbai-embed-large supports **Matryoshka Representation Learning (MRL)**[^2] -- leading dimensions carry the most information, so we truncate and re-normalize:

{:refdef: style="text-align: center;"}
$$\mathbf{v}_{d} = \frac{\mathbf{v}_{1:d}}{\|\mathbf{v}_{1:d}\|}$$
{: refdef}

Evaluated on 40 test queries (common words, rare phobias, scientific terms, abstract concepts):

| Dimensions | MRR | Hit@1 | Hit@6 |
|:---:|:---:|:---:|:---:|
| 1024 | 0.6733 | 24/40 | 32/40 |
| 768 | 0.6720 | 24/40 | 33/40 |
| 512 | 0.6645 | 23/40 | 33/40 |
| **384** | **0.6529** | **23/40** | **32/40** |
| 256 | 0.6590 | 23/40 | 32/40 |
| 128 | 0.5944 | 20/40 | 30/40 |

256d actually edges out 384d on MRR (0.6590 vs 0.6529) -- the Matryoshka ordering isn't strictly monotonic on a small test set. I went with 384d for the slightly better Hit@6 and to stay consistent with the query model's native output.

## Int8 scalar quantization

At 384d, float32 embeddings are 176 MB. Per-dimension scalar quantization to int8:

{:refdef: style="text-align: center;"}
$$q_d = \mathrm{round}\!\left(\frac{v_d - \min_d}{\max_d - \min_d} \times 255\right) \in [0, 255]$$
{: refdef}

At query time, the dot product decomposes into a scaled int8 dot product plus a per-query constant -- cheap to compute. Int8 at 384d: MRR 0.6483 vs float32's 0.6529, negligible loss. Final embedding file: **65 MB** for 176k entries.

## 1-bit binary quantization

Int8 is 4x smaller than float32, but we can go further. **Binary quantization** maps each dimension to a single bit (positive → 1, negative → 0), replacing cosine similarity with **Hamming distance** via XOR + popcount -- bitwise operations that modern CPUs execute in a single instruction.

The naive approach (threshold at zero) loses information because the dimension means aren't centered. **Iterative Quantization (ITQ)**[^4] learns an orthogonal rotation $R$ that minimizes quantization error:

{:refdef: style="text-align: center;"}
$$\mathbf{b} = \text{sign}\!\left((\mathbf{v} - \boldsymbol{\mu}) \cdot R\right)$$
{: refdef}

The rotation matrix is learned offline in ~50 iterations on a subsample of the definition embeddings. At query time, the query vector gets the same rotation before binarization.

**Does 384d matter for binary?** MRL concentrates semantic information in leading dimensions, but binary quantization is lossy -- more bits might help. I tested pure binary retrieval at different MRL truncations on a 3K-entry subsample (62 valid queries):

| Dimensions | Pure Binary MRR | Hit@1 |
|:---:|:---:|:---:|
| 1024 | 0.8022 | 50/62 |
| 768 | 0.8094 | 50/62 |
| **384** | **0.8210** | **51/62** |
| 192 | 0.7282 | 42/62 |

384d is optimal -- more dimensions past this point add noise rather than signal under binary quantization, consistent with the MRL property of concentrating information in leading dimensions.

**Two-stage binary + int8 reranking.** Pure binary retrieval at 384d is fast (~13ms for 176k entries) but loses some precision compared to int8. A two-stage pipeline recovers it: binary Hamming distance narrows from 176k to the top 500 candidates, then int8 dot product reranks them. On the 67-query test set this exactly matches pure int8 quality (MRR 0.6305 for both) while being significantly faster.

**Desktop** uses the two-stage pipeline (~8 MB binary + 65 MB int8). **Mobile** uses pure binary scoring only (~8 MB), keeping the total download under 30 MB -- a practical tradeoff since pure binary still achieves MRR 0.5782 (92% of int8 quality).

## The mobile problem

On iOS Safari, the 22M-parameter ONNX model fails to load -- both q8 and q4f16 quantizations produce `RangeError: Out of Memory`. WASM on iOS has strict memory limits, and the model's runtime footprint exceeds them. WebGPU isn't available on iOS either.

I evaluated four smaller transformer models in symmetric mode (same model for both queries and definitions, separate embedding files):

| Model | Params | Symmetric MRR | Hit@6 |
|-------|:---:|:---:|:---:|
| [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs){:target="_blank"} | 22.6M | 0.6296 | 29/40 |
| [bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2){:target="_blank"} | 17.4M | 0.5491 | 27/40 |
| [paraphrase-albert-small-v2](https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2){:target="_blank"} | 11.7M | 0.5392 | 26/40 |
| [paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2){:target="_blank"} | 17.4M | 0.5155 | 25/40 |

Snowflake leads, but at 22.6M params it has the same OOM problem. The rest offer only marginal gains over the static embedding approach I was already using.

## Lite mode: static embeddings in pure JavaScript

For devices where the ONNX model can't load, WordFor falls back to [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M){:target="_blank"} -- a [Model2Vec](https://github.com/MinishLab/model2vec){:target="_blank"} static embedding model. Inference is pure JavaScript:

1. WordPiece tokenize the query
2. Look up each token's 256d vector from a float16 matrix (~15 MB)
3. Mean pool and L2-normalize

Sub-1ms, zero dependencies. ~~Lite mode compensates for lower semantic quality with a keyword overlap score.~~ Quality weights computed at build time from cross-source agreement provide a soft re-ranking signal (see below).

## Fine-tuning the static model

The off-the-shelf potion-base-8M was trained on general text. WordFor's definitions have distinctive structure -- concise, noun-phrase-heavy, often starting with articles. Fine-tuning on in-domain data should help.

sentence-transformers' `StaticEmbedding` module wraps a Model2Vec model's embedding matrix as a differentiable `torch.nn.EmbeddingBag`[^3]. Training pipeline:

1. **Data**: [Open English WordNet 2025](https://en-word.net/){:target="_blank"} (CC-BY 4.0, 120k+ synsets) + [Webster's 1913](https://github.com/adambom/dictionary){:target="_blank"} (public domain). Merged and deduplicated with [SemHash](https://github.com/MinishLab/semhash){:target="_blank"} (threshold 0.9) to ~176k unique definitions.
2. **Pairs**: Each (word, definition) becomes a positive pair -- the word is the "query" and the definition is the "passage". ~845k training pairs.
3. **Loss**: `MultipleNegativesRankingLoss` with in-batch negatives -- for a batch of 512, each pair has 511 negatives for free.
4. **Decontamination**: SemHash cross-deduplication removes training definitions that are too similar (threshold 0.85) to the 40 evaluation queries. 12 entries removed in practice.

The embedding matrix is the only trainable parameter -- 30k tokens $\times$ 256 dimensions. No attention layers, no positional encodings, just a lookup table getting nudged by contrastive gradients. Training takes ~22 minutes on a single GPU.

| Version | Training data | MRR | Hit@1 | Hit@6 |
|---------|--------------|:---:|:---:|:---:|
| Baseline (off-the-shelf) | -- | 0.4859 | 16/40 | 24/40 |
| v1 | OEWN only | 0.5401 | 18/40 | 26/40 |
| **v4 (deployed)** | **OEWN + Webster's** | **0.5499** | **18/40** | **27/40** |

+13% MRR from fine-tuning alone. The runtime is unchanged -- same WordPiece tokenization, same mean-pooling, same sub-millisecond inference. Only the embedding matrix weights are different.

## Rust WASM for lite mode inference

The pure JavaScript Model2Vec implementation works, but it has a weakness: a hand-rolled WordPiece tokenizer that doesn't match the original HuggingFace tokenizer exactly. Edge cases in normalization, unknown token handling, and subword splitting can produce slightly different embeddings.

[model2vec-rs](https://github.com/MinishLab/model2vec-rs){:target="_blank"} is an official Rust implementation of Model2Vec that includes the full HuggingFace tokenizer via the [tokenizers](https://github.com/huggingface/tokenizers){:target="_blank"} crate. Compiling to WebAssembly gives us exact tokenizer parity with the Python training pipeline, in a 1.9 MB `.wasm` binary.

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

## Data-centric quality scoring

Embedding similarity alone has blind spots: compound words like "art teacher" outscore "teacher", and bag-of-words models confuse antonyms ("fear" vs "fearless"). Rather than engineering the ranking function, we improve the *data* -- a quality weight $q$ for every definition, computed at build time from four independent sources.

**Cross-source agreement.** Each word is checked against OEWN, Webster's, [Wiktionary](https://kaikki.org/){:target="_blank"} (1.3M headwords, CC-BY-SA 3.0), and [ConceptNet 5.7](https://conceptnet.io/){:target="_blank"} (878K English concepts, CC-BY-SA 4.0). Words confirmed by more sources get a small boost. Wiktionary and ConceptNet are used at build time only and not redistributed, so the output remains CC-BY 4.0.

**ConceptNet knowledge graph.** ConceptNet's 3M English edges provide two additional signals: *centrality* (log-normalized degree -- well-connected concepts like "water" score higher than obscure terms) and *relation diversity* (concepts participating in many relation types -- IsA, HasProperty, Synonym, etc. -- are better-defined).

**Definition quality.** Heuristics penalize short ($<$8 tokens), overly long ($>$25 tokens), and generic definitions ("a kind of ..."). An IDF-based richness score rewards definitions using specific vocabulary.

**Compound penalty.** Multi-word entries where all components exist as standalone words get penalized. The raw score is dampened via $q^{0.1}$ to act as a soft re-ranking signal rather than a hard filter:

{:refdef: style="text-align: center;"}
$$\text{score}_{\text{final}} = \text{cosine}(q_{\text{user}}, e_{\text{def}}) \cdot q^{0.1}$$
{: refdef}

On the expanded 67-query test set, quality weighting alone improved the base potion model by +7.8% MRR -- matching a fine-tuned model without changing any weights.

## Results

The full evaluation on a 67-query test set with near-miss pairs (active/passive semantics, antonyms, domain specificity):

**Full mode** (asymmetric: mdbr-leaf-mt queries, mxbai-embed-large definitions, 176k entries):

| Config | MRR | Hit@1 | Hit@6 |
|--------|:---:|:---:|:---:|
| **binary + int8 rerank (deployed desktop)** | **0.6305** | **36/67** | **54/67** |
| int8 only (384d) | 0.6305 | 36/67 | 54/67 |
| pure binary ITQ (deployed mobile) | 0.5782 | 32/67 | 51/67 |

**Lite mode** (symmetric: fine-tuned potion-base-8M, 256d):

| Config | MRR | Hit@1 | Hit@6 |
|--------|:---:|:---:|:---:|
| fine-tuned potion v4 + quality weights | 0.1248 | 4/67 | 14/67 |
| potion base (no quality weights) | 0.4368 | 22/67 | 38/67 |
| **potion base + quality weights (Wikt+ConceptNet)** | **0.4708** | **25/67** | **41/67** |

Note: full-mode and lite-mode evaluations use different embedding sets (asymmetric vs symmetric), so MRR values are not directly comparable across blocks.

### Browser benchmarks

Scoring latency over 176k entries (average of 5 queries):

| | Lite (Potion int8) | Full binary + int8 rerank | Full pure binary | Full int8 only |
|---|:---:|:---:|:---:|:---:|
| **Latency** | ~46 ms | ~46 ms | ~13 ms | ~76 ms |
| **Data size** | ~45 MB | ~76 MB | ~8 MB | ~68 MB |

Full mode also requires the mdbr-leaf-mt ONNX model (~22 MB, q8 WASM, ~50ms/query inference). Mobile uses pure binary to keep total download under 30 MB.

Deployed configuration: binary + int8 rerank on desktop, pure binary on mobile, automatic fallback to lite if the ONNX model fails to load.

## Architecture

```
    BUILD TIME (GPU)                    BROWSER (runtime)
    ----------------                    -----------------
    mxbai-embed-large-v1 (335M)        Full: mdbr-leaf-mt (22M) ONNX WASM
    -> 176k defs -> 1024d -> MRL 384d   -> query -> 384d
    -> ITQ binary (8 MB)                -> XOR+popcount 176k -> top 500
    -> int8 (65 MB, desktop only)       -> int8 rerank -> ~46ms (desktop)
                                        -> pure binary only -> ~13ms (mobile)

    fine-tuned potion (8M)             Lite: model2vec-rs WASM (or pure JS)
    -> 176k defs -> 256d -> int8        -> query -> 256d -> dot 176k x q
    -> 45 MB                            -> ~46ms

    Quality scoring (4 sources):
    OEWN + Webster + Wiktionary
    + ConceptNet -> per-entry q weight
```

## What makes it fast

- **1-bit binary scoring**: XOR + popcount over 176k entries in ~13ms -- 6x faster than int8 dot products
- **Two-stage retrieval**: binary first-pass narrows to 500 candidates, int8 rerank recovers full quality
- **Int8 quantization**: 4x smaller files than float32, cheap scaled dot product
- **Selective loading**: mobile skips the 65 MB int8 file entirely; full mode skips potion data
- **Service Worker caching**: cache-first after first visit
- **Model warmup**: dummy inference during loading avoids first-query compilation stall

## Privacy

Everything runs client-side. Static files served from GitHub Pages through Cloudflare CDN (see my post on [making GitHub Pages GDPR-compliant with Cloudflare](/github-pages-cloudflare-gdpr)). No server, no database, no cookies. Analytics via [GoatCounter](https://www.goatcounter.com/){:target="_blank"} -- cookie-free, no personal data.

## What's next

- **Model2Vec distillation**: Distilling from mxbai-embed-large-v1 could produce a static embedding matrix that better captures the teacher's space -- without changing runtime latency.
- **LLM-augmented definitions**: Using a large language model at build time to clean, expand, and rephrase definitions could improve embedding quality for rare or poorly-defined words.

## Try it

**[wordfor.xyz](https://wordfor.xyz){:target="_blank"}** -- describe a concept, find the word.

If you find it useful, consider a share or a [Ko-fi](https://ko-fi.com/zshn25){:target="_blank"}.

___

(c) Zeeshan Khan Suri, [<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc.svg" width="60"/>](http://creativecommons.org/licenses/by-nc/4.0/)

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
