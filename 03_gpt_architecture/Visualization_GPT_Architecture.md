# 📊 Step 03 — GPT Architecture: Visual Workflow

> **Companion visualization for** [`GPT_Model.ipynb`](file:///d:/projects/LLM/03_gpt_architecture/GPT_Model.ipynb), [`GPT_Model.md`](file:///d:/projects/LLM/03_gpt_architecture/GPT_Model.md), **and** [`Transformer_Block.md`](file:///d:/projects/LLM/03_gpt_architecture/Transformer_Block.md)

---

## 1. Full GPTModel Macro-Architecture

```mermaid
flowchart TD
    subgraph InputStage["📥 Input Processing"]
        TEXT["Input Text: 'every effort moves you'"]
        TEXT --> BPE["tiktoken BPE Tokenizer (gpt2)\nV = 50,257"]
        BPE --> IDS["Token ID Tensor: in_idx\nShape: [B, T]  ·  e.g. [2, 4]"]
    end

    subgraph EmbeddingStage["🧩 Embedding & Dropout"]
        IDS --> TOK["tok_emb: nn.Embedding(50257, 768)\nToken Embeddings → [B, T, 768]"]
        IDS --> POS["pos_emb: nn.Embedding(1024, 768)\nPositional Embeddings → [T, 768]"]
        TOK --> SUM["➕ Element-wise Sum\nx = tok_emb + pos_emb → [B, T, 768]"]
        POS --> SUM
        SUM --> DROP["drop_emb: nn.Dropout(0.1)\n→ [B, T, 768]"]
    end

    subgraph TransformerStack["🔁 N × Transformer Blocks (n_layers = 12)"]
        DROP --> TB1["TransformerBlock 1\n[B, T, 768] → [B, T, 768]"]
        TB1 --> TB2["TransformerBlock 2\n[B, T, 768] → [B, T, 768]"]
        TB2 --> DOTS["⋮ Blocks 3–11 ⋮"]
        DOTS --> TB12["TransformerBlock 12\n[B, T, 768] → [B, T, 768]"]
    end

    subgraph OutputStage["📤 Output Projection"]
        TB12 --> FNORM["final_norm: LayerNorm(768)\n→ [B, T, 768]"]
        FNORM --> HEAD["out_head: nn.Linear(768, 50257, bias=False)\n→ [B, T, 50257]"]
        HEAD --> LOGITS["Output Logits\nShape: [B, T, V] = [2, 4, 50257]"]
    end
```

---

## 2. Single Transformer Block (Pre-LN Design)

```mermaid
flowchart TD
    IN["Input X\n[B, T, 768]"] --> SKIP1["── Shortcut 1 ──"]
    IN --> N1["norm1: LayerNorm"]

    N1 --> ATT["att: MultiHeadAttention\n12 heads · d_k = 64"]
    ATT --> D1["drop_shortcut: Dropout(0.1)"]
    D1 --> ADD1["➕ Residual Add 1\nX₁ = shortcut + dropout(att(norm(X)))"]
    SKIP1 --> ADD1

    ADD1 --> SKIP2["── Shortcut 2 ──"]
    ADD1 --> N2["norm2: LayerNorm"]

    N2 --> FF["ff: FeedForward\nLinear(768→3072) → GELU → Linear(3072→768)"]
    FF --> D2["drop_shortcut: Dropout(0.1)"]
    D2 --> ADD2["➕ Residual Add 2\nX_out = shortcut + dropout(ff(norm(X₁)))"]
    SKIP2 --> ADD2

    ADD2 --> OUT["Output X_out\n[B, T, 768]"]
```

---

## 3. Multi-Head Attention (Head Split → Score → Merge)

```mermaid
flowchart TD
    X["Hidden State X\n[B, T, 768]"]

    X --> WQ["W_query: nn.Linear(768, 768)\n→ Q [B, T, 768]"]
    X --> WK["W_key: nn.Linear(768, 768)\n→ K [B, T, 768]"]
    X --> WV["W_value: nn.Linear(768, 768)\n→ V [B, T, 768]"]

    WQ --> SPLIT_Q["Reshape & Transpose\n[B, T, 12, 64] → [B, 12, T, 64]"]
    WK --> SPLIT_K["Reshape & Transpose\n[B, T, 12, 64] → [B, 12, T, 64]"]
    WV --> SPLIT_V["Reshape & Transpose\n[B, T, 12, 64] → [B, 12, T, 64]"]

    SPLIT_Q --> SCORES["Attention Scores\nQ @ Kᵀ → [B, 12, T, T]"]
    SPLIT_K --> SCORES

    SCORES --> MASK["Causal Mask\nUpper-triangle → -∞"]
    MASK --> SCALE["Scale: ÷ √64 = ÷ 8"]
    SCALE --> SOFT["Softmax (dim=-1)\nWeights A → [B, 12, T, T]"]

    SOFT --> DROP_A["Dropout on Weights"]
    DROP_A --> CTX["Context = A @ V\n→ [B, 12, T, 64]"]
    SPLIT_V --> CTX

    CTX --> MERGE["Transpose & Reshape\n[B, T, 12×64] → [B, T, 768]"]
    MERGE --> PROJ["out_proj: nn.Linear(768, 768)\n→ [B, T, 768]"]
```

---

## 4. Causal Mask Visualization

```
Causal (Autoregressive) Mask for T = 6 tokens:

         Key positions →
          t₁    t₂    t₃    t₄    t₅    t₆
     ┌─────────────────────────────────────────┐
 t₁  │  ✅    ❌    ❌    ❌    ❌    ❌   │   Token 1 sees only itself
 t₂  │  ✅    ✅    ❌    ❌    ❌    ❌   │   Token 2 sees tokens 1–2
Query t₃  │  ✅    ✅    ✅    ❌    ❌    ❌   │   Token 3 sees tokens 1–3
pos.  t₄  │  ✅    ✅    ✅    ✅    ❌    ❌   │   Token 4 sees tokens 1–4
 t₅  │  ✅    ✅    ✅    ✅    ✅    ❌   │   Token 5 sees tokens 1–5
 t₆  │  ✅    ✅    ✅    ✅    ✅    ✅   │   Token 6 sees tokens 1–6
     └─────────────────────────────────────────┘

✅ = Attend (score retained)
❌ = Masked (score set to -∞, softmax → 0.0)

Implementation: torch.triu(torch.ones(T, T), diagonal=1)
→ Upper triangle = 1 → masked_fill with -inf
```

---

## 5. Feed-Forward Network Expansion & Contraction

```mermaid
flowchart LR
    IN["Input\n[B, T, 768]"] --> L1["nn.Linear\n768 → 3072\n(4× expansion)"]
    L1 --> GELU["GELU\nActivation"]
    GELU --> L2["nn.Linear\n3072 → 768\n(contract back)"]
    L2 --> OUT["Output\n[B, T, 768]"]
```

```
Dimension trace through FeedForward:

    [B, T, 768]  ──Linear──▶  [B, T, 3072]  ──GELU──▶  [B, T, 3072]  ──Linear──▶  [B, T, 768]
         D                       4 × D                      4 × D                      D
```

---

## 6. GELU vs ReLU Activation

```
                          GELU vs ReLU Comparison
     ▲ output
   2 │                                          ╱  GELU
     │                                       ╱╱╱
   1 │                                    ╱╱╱
     │                                 ╱╱╱          ╱  ReLU
   0 │─────────────────────────────╱╱╱──────────────
     │                          ╱╱╱
  -1 │    GELU dips slightly ↙╱╱
     │                    ╱╱
  -2 │───────────────────────────────────────────▶ input
    -4        -2         0          2          4

Key: GELU is smooth everywhere (no hard kink at 0)
     → Avoids "dead neuron" problem of ReLU
     → Small non-zero gradients for negative inputs
```

---

## 7. Residual Connection Gradient Preservation

```mermaid
flowchart LR
    subgraph Without["❌ Without Shortcuts"]
        W1["Layer 0\nGrad: 0.000150"] --> W2["Layer 1\nGrad: 0.000140"]
        W2 --> W3["Layer 2\nGrad: 0.000607"]
        W3 --> W4["Layer 3\nGrad: 0.001125"]
        W4 --> W5["Layer 4\nGrad: 0.004503"]
    end

    subgraph With["✅ With Shortcuts"]
        S1["Layer 0\nGrad: 0.231017"] --> S2["Layer 1\nGrad: 0.237078"]
        S2 --> S3["Layer 2\nGrad: 0.348110"]
        S3 --> S4["Layer 3\nGrad: 0.133329"]
        S4 --> S5["Layer 4\nGrad: 1.821952"]
    end

    style With fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
```

---

## 8. GPT-2 124M Parameter Breakdown

```mermaid
pie title GPT-2 124M Parameter Distribution (163M raw · 124M unique with weight tying)
    "Token Embeddings (tok_emb)" : 38597376
    "Positional Embeddings (pos_emb)" : 786432
    "12 × Transformer Blocks" : 84971520
    "Final LayerNorm" : 1536
    "Output Head (out_head) — tied with tok_emb" : 38597376
```

```
Detailed Breakdown:
┌─────────────────────────────────────────────────────────┐
│ Component                    │ Params        │ % Total  │
├─────────────────────────────────────────────────────────┤
│ tok_emb: [50257 × 768]      │ 38,597,376    │ 23.7%    │
│ pos_emb: [1024 × 768]       │    786,432    │  0.5%    │
│ 12 Transformer Blocks:       │               │          │
│   ├─ MHA (W_q,W_k,W_v,W_o) │ 2,359,296 ×12 │ 17.4%    │
│   ├─ FFN (L1 + L2)          │ 4,718,592 ×12 │ 34.7%    │
│   └─ 2 × LayerNorm          │     3,072 ×12 │  0.02%   │
│ final_norm                   │     1,536     │  ~0%     │
│ out_head: [768 × 50257]     │ 38,597,376    │ 23.7%    │
├─────────────────────────────────────────────────────────┤
│ Raw Total                    │ 163,009,536   │ 100%     │
│ With Weight Tying (unique)   │ 124,412,160   │ ~124M    │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Complete Tensor Shape Trace (One Forward Pass)

| Layer | Input Shape | Operation | Output Shape |
|---|:---:|---|:---:|
| `in_idx` | — | Token ID tensor | `[B, T]` |
| `tok_emb` | `[B, T]` | Embedding lookup | `[B, T, 768]` |
| `pos_emb` | `[T]` | Position embedding | `[T, 768]` |
| Sum | `[B,T,768]` + `[T,768]` | Broadcast add | `[B, T, 768]` |
| `drop_emb` | `[B, T, 768]` | Dropout | `[B, T, 768]` |
| **TransformerBlock ×12** | | | |
| → `norm1` | `[B, T, 768]` | LayerNorm | `[B, T, 768]` |
| → Q, K, V | `[B, T, 768]` | 3× Linear | `[B, T, 768]` each |
| → Head split | `[B, T, 768]` | Reshape+Transpose | `[B, 12, T, 64]` |
| → Scores | `[B,12,T,64]` | Q @ Kᵀ | `[B, 12, T, T]` |
| → Mask+Scale+Softmax | `[B,12,T,T]` | Causal attention | `[B, 12, T, T]` |
| → Context | `[B,12,T,T]` × `[B,12,T,64]` | A @ V | `[B, 12, T, 64]` |
| → Merge | `[B,12,T,64]` | Transpose+Reshape | `[B, T, 768]` |
| → `out_proj` | `[B, T, 768]` | Linear | `[B, T, 768]` |
| → Residual 1 | `[B,T,768]` + `[B,T,768]` | Add shortcut | `[B, T, 768]` |
| → `norm2` | `[B, T, 768]` | LayerNorm | `[B, T, 768]` |
| → FFN Linear 1 | `[B, T, 768]` | 768→3072 | `[B, T, 3072]` |
| → GELU | `[B, T, 3072]` | Activation | `[B, T, 3072]` |
| → FFN Linear 2 | `[B, T, 3072]` | 3072→768 | `[B, T, 768]` |
| → Residual 2 | `[B,T,768]` + `[B,T,768]` | Add shortcut | `[B, T, 768]` |
| `final_norm` | `[B, T, 768]` | LayerNorm | `[B, T, 768]` |
| `out_head` | `[B, T, 768]` | Linear 768→50257 | **`[B, T, 50257]`** |
