# Transformer Block Architecture Explained

Based on `code/Tranformer_Block.ipynb` and `code/Multiheadattention.py`, this document explains the full architecture of a standard **Transformer Block** used in GPT models.

---

## 1. Overview of the Transformer Block

A Transformer Block combines **Multi-Head Causal Self-Attention**, **Layer Normalization**, **Feed-Forward Networks (FFN)** with GELU activations, and **Residual (Shortcut) Connections** with dropout.

```
       Input x
          │
          ├───┐ (Shortcut Connection)
          │   ▼
     LayerNorm(x)
          │
  MultiHeadAttention
          │
       Dropout
          │
          ▼
   (+) ◄──┴── x + Attention Output
    │
    ├───┐ (Shortcut Connection)
    │   ▼
  LayerNorm
    │
   FeedForward (Linear -> GELU -> Linear)
    │
 Secondary Dropout
    │
    ▼
   (+) ◄──┴── Output Tensor
```

---

## 2. Component Details

### 2.1 Multi-Head Attention (`MultiHeadAttention`)
Implemented in [`code/Multiheadattention.py`](file:///d:/projects/LLM/code/Multiheadattention.py):
- **Query, Key, Value Projections**: Projects input tensor of dimension `d_in` into `d_out`.
- **Head Splitting**: Reshapes projections into `[batch, num_tokens, num_heads, head_dim]`.
- **Causal Masking**: Uses an upper-triangular mask (`torch.triu(..., diagonal=1)`) to ensure tokens only attend to past and current positions.
- **Scaled Dot-Product Attention**:
  $$\text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- **Concatenation & Output Projection**: Combines all heads back into `[batch, num_tokens, d_out]` and applies a linear projection (`out_proj`).

---

### 2.2 Transformer Block Implementation (`TransformerBlock`)

The `TransformerBlock` module integrates all preceding components:

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 1. First residual block: LayerNorm -> Multi-Head Attention -> Dropout -> Add
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 2. Second residual block: LayerNorm -> FeedForward -> Dropout -> Add
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
```

---

## 3. Shape Invariance

One key design principle of the `TransformerBlock` is **shape preservation**:
$$\text{Input Shape } [B, T, D] \xrightarrow{\quad\text{TransformerBlock}\quad} \text{Output Shape } [B, T, D]$$

- **Batch Size ($B$)**: Number of sequences in the batch.
- **Sequence Length ($T$)**: Number of tokens per sequence (up to `context_length`).
- **Embedding Dimension ($D$)**: Feature dimension size `emb_dim` (e.g. 768 for GPT-2 124M).

Because the output shape matches the input shape exactly, multiple `TransformerBlock` instances can be stacked sequentially to build deep GPT architectures.
