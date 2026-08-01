# 📊 Step 02 — Attention Mechanism: Visual Workflow

> **Companion visualization for** [`Attention_Mechanism.ipynb`](file:///d:/projects/LLM/02_attention_mechanism/Attention_Mechanism.ipynb) **and** [`Attention_Mechanism.md`](file:///d:/projects/LLM/02_attention_mechanism/Attention_Mechanism.md)

---

## 1. Self-Attention Evolution Roadmap

```mermaid
flowchart LR
    subgraph V0["Level 0 · Conceptual Baseline"]
        A0["Simple Self-Attention\nX acts as Q, K, V\nNo trainable weights"]
    end

    subgraph V1["Level 1 · Vectorized"]
        A1["Matrix Self-Attention\nS = X @ Xᵀ\nA = softmax(S)\nZ = A @ X"]
    end

    subgraph V2["Level 2 · Trainable Weights"]
        A2["QKV Projections\nQ = X·W_q  K = X·W_k  V = X·W_v\nScaled: S / √d_k\nnn.Parameter"]
    end

    subgraph V3["Level 3 · Production Module"]
        A3["SelfAttention_v2\nnn.Linear projections\nOptional bias\nReady for MultiHead"]
    end

    A0 --> A1 --> A2 --> A3

    style V3 fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
```

---

## 2. Scaled Dot-Product Self-Attention (Full Workflow)

```mermaid
flowchart TD
    subgraph Input["📥 Input"]
        X["Input Embeddings: X\nShape: [N, d_in]\ne.g. [6, 3]"]
    end

    subgraph Projection["🔀 Linear Projections"]
        X -->|"X @ W_q"| Q["Query: Q\n[N, d_out]"]
        X -->|"X @ W_k"| K["Key: K\n[N, d_out]"]
        X -->|"X @ W_v"| V["Value: V\n[N, d_out]"]
    end

    subgraph Scoring["📐 Scoring"]
        Q --> DOT["Raw Scores\nS = Q @ Kᵀ\n[N, N]"]
        K --> DOT
        DOT --> SCALE["Scaled Scores\nS / √d_k\n[N, N]"]
    end

    subgraph Normalization["📊 Normalization"]
        SCALE --> SOFTMAX["Attention Weights\nA = softmax(S_scaled, dim=-1)\n[N, N]\nEach row sums to 1.0"]
    end

    subgraph Aggregation["🎯 Aggregation"]
        SOFTMAX --> MATMUL["Context Vectors\nZ = A @ V\n[N, d_out]"]
        V --> MATMUL
    end
```

---

## 3. Single-Token Attention Walkthrough (Token 2 as Query)

```
6 Tokens, 3-dim embeddings:

Token 1 ("Your")    = [0.43, 0.15, 0.89]
Token 2 ("journey") = [0.55, 0.87, 0.66]  ← Query
Token 3 ("starts")  = [0.57, 0.85, 0.64]
Token 4 ("with")    = [0.22, 0.58, 0.33]
Token 5 ("one")     = [0.77, 0.25, 0.10]
Token 6 ("step")    = [0.05, 0.80, 0.55]

Step 1: Dot-Product Similarity (query = Token 2)
┌──────────────────────────────────────────────────────┐
│  score(2,1) = dot([0.55,0.87,0.66], [0.43,0.15,0.89])  │
│  score(2,2) = dot([0.55,0.87,0.66], [0.55,0.87,0.66])  │  ← self-similarity (highest)
│  score(2,3) = dot([0.55,0.87,0.66], [0.57,0.85,0.64])  │
│  ...                                                    │
└──────────────────────────────────────────────────────┘

Step 2: Softmax Normalization
┌──────────────────────────────────────────────────────┐
│  α₂ = softmax([score₁, score₂, ..., score₆])         │
│  Sum of weights = 1.0                                 │
└──────────────────────────────────────────────────────┘

Step 3: Weighted Aggregation
┌──────────────────────────────────────────────────────┐
│  z₂ = α₂₁·x₁ + α₂₂·x₂ + α₂₃·x₃ + ... + α₂₆·x₆    │
│  → Context vector for Token 2: [3-dim vector]         │
└──────────────────────────────────────────────────────┘
```

---

## 4. Attention Weight Matrix Heatmap (Conceptual)

```
                    Keys (K)
            Token1  Token2  Token3  Token4  Token5  Token6
         ┌────────────────────────────────────────────────┐
Token1   │  0.22    0.18    0.17    0.14    0.16    0.13  │  → sum = 1.0
Token2   │  0.15    0.21    0.20    0.14    0.13    0.17  │  → sum = 1.0
Queries  Token3   │  0.16    0.20    0.20    0.14    0.14    0.16  │  → sum = 1.0
(Q)      Token4   │  0.14    0.18    0.18    0.16    0.15    0.19  │  → sum = 1.0
Token5   │  0.19    0.16    0.16    0.13    0.22    0.14  │  → sum = 1.0
Token6   │  0.13    0.19    0.19    0.16    0.13    0.20  │  → sum = 1.0
         └────────────────────────────────────────────────┘
                    Attention Weight Matrix A [N, N]

Each cell A[i,j] = "how much Token i attends to Token j"
Higher value = stronger influence from Token j on Token i's output
```

---

## 5. Tensor Shape Transformation Pipeline

```mermaid
flowchart TD
    X["Input X\n[N, d_in] = [6, 3]"]

    X --> WQ["W_query\n[d_in, d_out] = [3, 2]"]
    X --> WK["W_key\n[d_in, d_out] = [3, 2]"]
    X --> WV["W_value\n[d_in, d_out] = [3, 2]"]

    WQ --> Q["Q = X @ W_q\n[6, 2]"]
    WK --> K["K = X @ W_k\n[6, 2]"]
    WV --> V["V = X @ W_v\n[6, 2]"]

    Q --> S["S = Q @ Kᵀ\n[6, 6]"]
    K --> S

    S --> SCALE["S_scaled = S / √2\n[6, 6]"]
    SCALE --> A["A = softmax(S_scaled)\n[6, 6]"]

    A --> Z["Z = A @ V\n[6, 2]"]
    V --> Z

    Z --> OUT["Output Context Vectors\n[N, d_out] = [6, 2]"]

    style OUT fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
```

---

## 6. PyTorch Module Comparison

```mermaid
flowchart TD
    subgraph V1["SelfAttention_v1 (nn.Parameter)"]
        P1["W_q = nn.Parameter(torch.rand(d_in, d_out))"]
        P2["W_k = nn.Parameter(torch.rand(d_in, d_out))"]
        P3["W_v = nn.Parameter(torch.rand(d_in, d_out))"]
        P1 --> F1["queries = x @ self.w_query"]
        P2 --> F1
        P3 --> F1
    end

    subgraph V2["SelfAttention_v2 (nn.Linear) ✅ Production"]
        L1["W_query = nn.Linear(d_in, d_out, bias=False)"]
        L2["W_key = nn.Linear(d_in, d_out, bias=False)"]
        L3["W_value = nn.Linear(d_in, d_out, bias=False)"]
        L1 --> F2["queries = self.W_query(x)"]
        L2 --> F2
        L3 --> F2
    end

    style V2 fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
```

---

## 7. Why Scale by $\frac{1}{\sqrt{d_k}}$?

```
Without Scaling (d_k = 64):
┌──────────────────────────────────────────────────┐
│ Raw scores S = Q @ Kᵀ  can have large magnitudes │
│ e.g. values like [45.2, 38.1, 2.3, 0.1, ...]    │
│                                                   │
│ softmax([45.2, 38.1, 2.3, 0.1, ...])             │
│ = [0.999, 0.001, 0.000, 0.000, ...]              │
│ → Nearly one-hot! Gradients vanish for all but 1  │
└──────────────────────────────────────────────────┘

With Scaling (÷ √64 = ÷ 8):
┌──────────────────────────────────────────────────┐
│ Scaled scores S / √d_k                            │
│ e.g. values like [5.65, 4.76, 0.29, 0.01, ...]  │
│                                                   │
│ softmax([5.65, 4.76, 0.29, 0.01, ...])           │
│ = [0.55, 0.37, 0.04, 0.03, ...]                  │
│ → Smooth distribution! Healthy gradient flow      │
└──────────────────────────────────────────────────┘
```

---

## 8. Summary Shape Table

| Step | Operation | Shape |
|:---:|---|:---:|
| Input | Token Embeddings $X$ | `[N, d_in]` |
| Projection | $Q = X W_q$ | `[N, d_out]` |
| Projection | $K = X W_k$ | `[N, d_out]` |
| Projection | $V = X W_v$ | `[N, d_out]` |
| Raw Scores | $S = Q K^\top$ | `[N, N]` |
| Scaled Scores | $S / \sqrt{d_k}$ | `[N, N]` |
| Attention Weights | $A = \text{softmax}(S_{\text{scaled}})$ | `[N, N]` |
| Context Vectors | $Z = A V$ | **`[N, d_out]`** |
