# nano-llm: understanding notes

*working notes alongside the implementation — understanding first, code second*

---

## 1. why this architecture?

The transformer (Vaswani et al., 2017) replaced recurrence with attention. The key shift:
RNNs process tokens sequentially, so gradients have to travel long paths through time to
connect distant tokens — they vanish. Transformers compute relationships between all tokens
in parallel, in a single pass.

Every component has a job. If you remove it, something specific breaks.

**token embeddings**
Discrete tokens (characters, subwords) can't be fed to a neural network directly. The
embedding matrix maps each token ID to a learned vector in R^d. The model operates entirely
in this continuous space. Size: vocab_size × d.

**positional encoding**
Attention is permutation-invariant — if you shuffle the input sequence, you get the same
result. That's a problem for language, where order is meaning. Positional encodings inject
position information into each embedding. Can be fixed (sinusoidal, as in the original paper)
or learned (GPT-style). We use learned — simpler, works fine at small scale.

**self-attention (the core)**
For each token, compute three vectors: Query (what am I looking for?), Key (what do I
contain?), Value (what do I contribute?). Attention score between token i and j:

    score(i, j) = dot(Q_i, K_j) / sqrt(d_k)

Scaled by sqrt(d_k) to prevent the dot products from growing large and saturating softmax.
Softmax over scores → weights → weighted sum of Values. Each token's output is a
context-aware mixture of all other tokens' values.

**causal masking**
For language modeling, we're predicting the next token — so token i must only attend to
tokens 0..i, never to future tokens. Implemented as an upper-triangular mask (set future
positions to -inf before softmax → they become 0 after softmax).

**multi-head attention**
Run attention h times in parallel with different learned projections. Each head can specialize:
one might track syntax, another coreference, another positional relationships. Outputs are
concatenated and projected back to d. Cost: same as single-head attention (projections split
d across heads, so d_k = d / h).

**residual connections**
Every sublayer is wrapped: output = x + sublayer(x). This creates a direct path for
gradients to flow back through the network without passing through any nonlinearity. Without
this, training deep networks is nearly impossible — gradients vanish or explode. Introduced
in ResNets (He et al., 2015), critical here too.

**layer normalization**
Normalize activations to zero mean and unit variance, then apply learned scale and shift.
Stabilizes training. GPT uses pre-norm (normalize before the sublayer, not after) — more
stable for deep networks, easier to train.

**feed-forward network (FFN)**
Two linear layers with a GELU nonlinearity between them. Applied independently to each
position. Expands d → 4d → d. If attention is "routing" (which tokens talk to which),
FFN is "computation" (what to do with the information once routed). Together they form
the full transformer block.

**weight tying**
The input embedding matrix (vocab_size × d) and the output projection (d → vocab_size) share
weights. Intuition: the vector that represents a token as input should be similar to the
vector that scores it as output. Halves the parameter count at the vocab interface, which
matters when vocab is large. Not critical at char-level but worth understanding.

---

## 2. the hardware layer

**CPU vs GPU**
CPUs: few powerful cores (8–32), high clock speed, optimized for sequential logic and
branch-heavy code. GPUs: thousands of simpler cores (thousands to tens of thousands),
optimized for SIMD (same instruction, multiple data). Matrix multiplication is
embarrassingly parallel — a (m×k) × (k×n) matmul requires m×n×k multiplications and
additions, all independent. GPUs dominate ML because of this.

**memory hierarchy**
Registers → L1 cache → L2 cache → L3 cache → DRAM → SSD. Latency increases ~10x at
each level. GPU compute is fast enough that memory bandwidth is often the bottleneck, not
FLOPs. Moving data between levels is expensive.

**discrete GPU (e.g., Colab T4)**
Separate VRAM (16GB on T4), connected to CPU via PCIe. Data must be explicitly transferred:
CPU RAM → PCIe → VRAM before GPU can touch it. PCIe bandwidth (~16 GB/s) is a real
bottleneck for data-heavy pipelines.

**Apple Silicon (M1/M2) — unified memory**
CPU and GPU share the same physical memory pool — no PCIe transfer. The GPU has direct
access to whatever the CPU allocated. This blurs the CPU/GPU line significantly: you can
have a 16GB M2 where the GPU can use all 16GB, vs a system with 32GB CPU RAM and a
discrete GPU with only 8GB VRAM. MPS (Metal Performance Shaders) is the PyTorch backend
for Apple Silicon (`device = "mps"`).

**float32 vs float16 vs bfloat16**

| dtype   | bytes | exponent bits | mantissa bits | notes                          |
|---------|-------|---------------|---------------|--------------------------------|
| fp32    | 4     | 8             | 23            | standard, full precision       |
| fp16    | 2     | 5             | 10            | limited range, can overflow    |
| bf16    | 2     | 8             | 7             | same range as fp32, less prec  |

bf16 is preferred for training — same dynamic range as fp32 so it doesn't overflow on
large activations/gradients, just slightly less precise. fp16 requires loss scaling to
avoid underflow in gradients. Most modern GPU training uses bf16 or mixed precision
(bf16 compute, fp32 accumulation).

---

## 3. the memory calculation

This is the calculation that separates "I trained a model" from "I understand what's
happening." Given a model architecture, you should be able to derive the exact memory
requirement and back-solve for the maximum batch size before touching any code.

**parameter count (small GPT)**

Hyperparameters:
- `V` = vocab size (65 for char-level on arXiv abstracts)
- `T` = context length / block size (e.g., 256)
- `d` = embedding dimension (e.g., 256)
- `L` = number of transformer layers (e.g., 6)
- `h` = number of attention heads (e.g., 8)

Per transformer block:
- LayerNorm 1: 2d (weight + bias)
- Attention (Q, K, V projections + output): 3(d²) + d² = 4d²  [ignoring biases]
- LayerNorm 2: 2d
- FFN (d→4d→d): 4d² + 4d² = 8d²  [ignoring biases]
- Total per block: ≈ 12d²

Full model:
- Token embedding: V × d
- Position embedding: T × d
- L blocks: L × 12d²
- Final LayerNorm: 2d
- Output projection: tied with token embedding → 0 extra

For d=256, L=6, V=65, T=256:
- Token emb: 65 × 256 = 16,640
- Pos emb: 256 × 256 = 65,536
- 6 blocks: 6 × 12 × 256² = 4,718,592
- Total ≈ 4.8M parameters

**memory for training (fp32)**

| component          | multiplier | size for 4.8M params |
|--------------------|------------|----------------------|
| parameters         | 1×         | 19.2 MB              |
| gradients          | 1×         | 19.2 MB              |
| AdamW m (momentum) | 1×         | 19.2 MB              |
| AdamW v (variance) | 1×         | 19.2 MB              |
| total              | 4×         | 76.8 MB              |

AdamW stores two fp32 optimizer state tensors per parameter (m and v), regardless of
training dtype. This is often the largest fixed cost for bigger models.

**activations (batch-size-dependent)**

Activations are stored during the forward pass for use in backprop. Per transformer block,
per sample in the batch:
- Attention scores: T × T
- Attention output: T × d
- FFN intermediate: T × 4d
- Rough total per block: T × (T + 5d) × bytes_per_element

For T=256, d=256, fp32 (4 bytes), L=6 blocks:
    per_sample = 6 × 256 × (256 + 5×256) × 4 bytes
               = 6 × 256 × 1536 × 4
               ≈ 9.4 MB per sample in batch

**back-solving for batch size**

    available_memory = total_device_memory × 0.8   (leave headroom)
    model_memory = 76.8 MB  (fixed)
    batch_memory = available_memory - model_memory
    max_batch_size = floor(batch_memory / per_sample_activation_cost)
    batch_size = largest power of 2 ≤ max_batch_size

Why powers of 2? GPU memory is physically organized in powers of 2. Tensors aligned to
powers of 2 map cleanly to memory banks — no padding waste, optimal memory access patterns,
maximum CUDA kernel efficiency. An odd batch size doesn't crash anything; it just leaves
compute on the table.

Example for M2 Mac (16GB unified, ~12GB usable for ML):
    batch_memory = 12,000 - 76.8 ≈ 11,923 MB
    max_batch_size = floor(11,923 / 9.4) ≈ 1268
    → batch_size = 1024  (nearest power of 2 ≤ 1268)

Example for Colab T4 (16GB VRAM):
    Similar — but VRAM, not unified. If model is loaded in fp16 instead, model_memory ≈ 38.4 MB
    and per_sample cost halves → can fit more.

The interviewer's point: don't say "I picked 32 because that's what the tutorial used."
Be able to derive the number, explain why it's a power of 2, and know what you'd do if
you ran out of memory (reduce batch size by half, try gradient checkpointing, switch dtype).

---

## 4. what went wrong

*(to be filled in during actual training runs)*

Placeholders for known common failure modes:

- **loss doesn't decrease**: learning rate too high (exploding gradients) or too low.
  Check: gradient norms, loss curve shape.
- **NaN loss**: overflow, often from fp16 without loss scaling, or from attention scores
  growing too large before scaling by sqrt(d_k).
- **loss decreases then plateaus**: learning rate needs decay, or model is underfitting
  (too small) / overfitting (too large, need more data).
- **CUDA OOM**: batch size too large. Halve it and retry. Add gradient checkpointing if
  still OOM (trades compute for memory by recomputing activations during backward pass).
- **slow training**: not using the right device. Check `next(model.parameters()).device`.

---

## 5. the corpus

arXiv ML/NLP abstracts (cs.LG + cs.CL), fetched via the arXiv API. ~5000 abstracts,
one per line in `data/corpus.txt`. Char-level tokenization — the vocabulary is just the
set of unique characters in the corpus.

Why char-level? It forces the model to learn everything from scratch: how to spell words,
what punctuation patterns look like, how sentences are structured. Nothing is pre-baked
into the vocabulary. Harder to train, but the learning is more transparent.

Why these abstracts? The writing is highly structured and repetitive in interesting ways:
"We propose...", "In this paper...", "Our method achieves state-of-the-art...". A model
trained on this will generate plausible-sounding but meaningless academic prose. The outputs
are funny and also reveal what statistical patterns the model has actually captured.

Expected character vocabulary: ~70–80 characters (lowercase + uppercase letters, digits,
punctuation, LaTeX fragments like \alpha or $).
