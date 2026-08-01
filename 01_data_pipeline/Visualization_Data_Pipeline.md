# 📊 Step 01 — Data Input Pipeline: Visual Workflow

> **Companion visualization for** [`Data_Input_Pipeline.ipynb`](file:///d:/projects/LLM/01_data_pipeline/Data_Input_Pipeline.ipynb) **and** [`Data_Input_Pipeline.md`](file:///d:/projects/LLM/01_data_pipeline/Data_Input_Pipeline.md)

---

## 1. End-to-End Data Pipeline Flowchart

```mermaid
flowchart TD
    subgraph Stage1["🔤 Stage 1 · Raw Text Ingestion"]
        A["📄 Raw Text File: verdict.txt\n(~20,643 characters)"]
        A --> B["Character Stream\nloaded into memory via open()"]
    end

    subgraph Stage2["✂️ Stage 2 · Tokenization"]
        B --> C1["Option A: Regex Split\nre.split on punctuation & whitespace"]
        B --> C2["Option B: NLTK\nword_tokenize()"]
        B --> C3["Option C: BPE (tiktoken)\nGPT-2 Encoding · V = 50,257"]
        C1 --> D["Token String List"]
        C2 --> D
        C3 --> E["Subword Token ID List"]
    end

    subgraph Stage3["🗂️ Stage 3 · Vocabulary & Special Tokens"]
        D --> F["Build Vocabulary Dict\nsorted unique tokens → integer IDs"]
        F --> G["Add Special Tokens\n‹|unk|› and ‹|endoftext|›"]
        G --> H["Encode: string → ID list\nDecode: ID list → string"]
        E --> H
    end

    subgraph Stage4["📐 Stage 4 · Sliding Window Sampling"]
        H --> I["GPTDatasetV1\n(max_length, stride)"]
        I --> J["Input x = tokens[i : i+L]\nTarget y = tokens[i+1 : i+L+1]"]
    end

    subgraph Stage5["📦 Stage 5 · DataLoader & Embeddings"]
        J --> K["PyTorch DataLoader\nbatch_size, shuffle, drop_last"]
        K --> L["Token Embedding\nnn.Embedding(V, D)"]
        K --> M["Positional Embedding\nnn.Embedding(T, D)"]
        L --> N["➕ Element-wise Sum\nX = E_token + E_pos"]
        M --> N
        N --> O["Final Input Tensor\nShape: [B, T, D]"]
    end
```

---

## 2. Tokenization Strategy Comparison

```mermaid
flowchart LR
    Raw["Raw Text\n'Hello, world!'"] --> R["Regex Split"]
    Raw --> N["NLTK Tokenizer"]
    Raw --> BPE["BPE (tiktoken)"]

    R --> R_Out["['Hello', ',', 'world', '!']\nCustom Vocab · OOV → ‹|unk|›"]
    N --> N_Out["['Hello', ',', 'world', '!']\nNLTK Vocab · OOV → ‹|unk|›"]
    BPE --> BPE_Out["[15496, 11, 995, 0]\nGPT-2 Vocab · V=50,257\nNo OOV — subword fallback"]

    style BPE fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    style BPE_Out fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
```

---

## 3. Sliding Window Dataset Visualization

```
Token IDs:  [ 50,  51,  52,  53,  54,  55,  56,  57 ]
             ─────────────────────────────────────────
max_length = 4, stride = 1

┌─── Sample 1 ───┐
│ Input  x: [ 50, 51, 52, 53 ]     Target y: [ 51, 52, 53, 54 ]  │
└────────────────┘
    ┌─── Sample 2 ───┐
    │ Input  x: [ 51, 52, 53, 54 ]  Target y: [ 52, 53, 54, 55 ]  │
    └────────────────┘
        ┌─── Sample 3 ───┐
        │ Input  x: [ 52, 53, 54, 55 ]  Target y: [ 53, 54, 55, 56 ]  │
        └────────────────┘
            ┌─── Sample 4 ───┐
            │ Input  x: [ 53, 54, 55, 56 ]  Target y: [ 54, 55, 56, 57 ]  │
            └────────────────┘

Each sample:  x[i] → y[i] = x[i] shifted right by 1 position
```

---

## 4. Tensor Shape Transformation Pipeline

```mermaid
flowchart TD
    A["Token ID Tensor (from DataLoader)\nShape: [B, T]  ·  e.g. [2, 4]"]

    A --> B["Token Embedding Layer\nnn.Embedding(50257, 768)"]
    B --> C["Token Embeddings\nShape: [B, T, D] → [2, 4, 768]"]

    A --> D["Positional Embedding Layer\nnn.Embedding(T, 768)\ntorch.arange(T)"]
    D --> E["Positional Embeddings\nShape: [T, D] → [4, 768]\nBroadcast to [1, T, D]"]

    C --> F["➕ Element-wise Addition"]
    E --> F

    F --> G["Final Input Embedding X\nShape: [B, T, D] → [2, 4, 768]"]
```

---

## 5. Embedding Summation Detail

```
Token Embedding Matrix E_token ∈ ℝ^{V × D}     (50,257 × 768)
                    │
                    ▼  Lookup by token IDs → [B, T, D]
┌───────────────────────────────────────────────┐
│  Batch 1, Token 1:  [0.12, -0.34, ..., 0.56] │  ← 768-dim vector
│  Batch 1, Token 2:  [0.78,  0.23, ..., -0.11]│
│  Batch 1, Token 3:  ...                       │
│  Batch 1, Token 4:  ...                       │
│  ─────────────────────────────────────────── │
│  Batch 2, Token 1:  ...                       │
│  ...                                          │
└───────────────────────────────────────────────┘

                    +

Positional Embedding Matrix E_pos ∈ ℝ^{T × D}  (4 × 768)
┌───────────────────────────────────────────────┐
│  Position 0:  [0.01,  0.05, ..., -0.03]       │  ← same for every batch
│  Position 1:  [0.08, -0.02, ...,  0.14]       │
│  Position 2:  ...                              │
│  Position 3:  ...                              │
└───────────────────────────────────────────────┘

                    =

Final Input X ∈ ℝ^{B × T × D}                   [2, 4, 768]
        → Ready for Attention layers
```

---

## 6. Summary Shape Table

| Stage | Component | Input Shape | Output Shape |
|:---:|---|:---:|:---:|
| 1 | Raw Text File | — | String (20,643 chars) |
| 2 | BPE Tokenizer (`tiktoken`) | String | Integer ID List (~5,145 IDs) |
| 3 | `GPTDatasetV1` Sliding Window | ID List | $(x, y)$ pairs, each `[T]` |
| 4 | `DataLoader` Batching | `[T]` pairs | `[B, T]` tensors |
| 5 | Token Embedding `nn.Embedding(V, D)` | `[B, T]` | `[B, T, D]` |
| 6 | Positional Embedding `nn.Embedding(T, D)` | `[T]` | `[T, D]` → broadcast `[1, T, D]` |
| 7 | Element-wise Sum `tok_emb + pos_emb` | `[B, T, D]` + `[1, T, D]` | **`[B, T, D]`** |
