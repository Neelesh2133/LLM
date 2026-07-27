# GPT Model Architecture Explained

Based on the `code/GPT_Model.ipynb` file, this document provides a detailed breakdown of the architecture for a GPT model, named `DummyGPTModel`, its configuration parameters, tokenization pipeline, model output shapes, and Layer Normalization concepts.

---

## 1. `DummyGPTModel` Architecture

The core component is a PyTorch module `DummyGPTModel` (`nn.Module`) which mirrors the macro-structure of a standard GPT architecture (e.g., GPT-2). It relies on a configuration dictionary (`cfg`) to initialize all inner layers.

### Key Components:
- **Token Embeddings (`tok_emb`)**: 
  - `nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])`
  - Maps token IDs into continuous vector representations of dimension `emb_dim`.
- **Positional Embeddings (`pos_emb`)**: 
  - `nn.Embedding(cfg["context_length"], cfg["emb_dim"])`
  - Learned positional embedding layer that encodes the position of each token in the sequence (up to `context_length`).
- **Dropout (`drop_emb`)**: 
  - `nn.Dropout(cfg["drop_rate"])`
  - Applied to the combined (token + positional) embeddings to prevent overfitting during training.
- **Transformer Blocks (`trf_blocks`)**: 
  - `nn.Sequential` containing `n_layers` instances of `DummyTransformerBlock`.
  - Placeholder modules for attention and feed-forward blocks.
- **Final Layer Normalization (`final_norm`)**: 
  - `DummyLayerNorm(cfg["emb_dim"])`
  - Normalizes features before passing to the final linear layer.
- **Output Head (`out_head`)**: 
  - `nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)`
  - Projects hidden states back into the vocabulary space to produce logits for next-token prediction.

---

## 2. Model Configuration (`GPT_CONFIG_124M`)

The notebook defines a configuration dictionary representing a 124 million parameter GPT-2 model baseline:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size (BPE tokens)
    "context_length": 1024,   # Maximum sequence context length
    "emb_dim": 768,           # Vector dimension of embeddings
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of transformer block layers
    "drop_rate": 0.1,         # Dropout probability rate
    "qkv_bias": False         # Query-Key-Value projection bias flag
}
```

---

## 3. Tokenization & Batch Preparation

The input text is tokenized and batched using the `tiktoken` library with the BPE (Byte Pair Encoding) `gpt2` tokenizer:

1. **Input Sentences**:
   - `"every effort moves you"`
   - `"every day holds a"`
2. **Encoding**:
   - Converted into token IDs using `tokenizer.encode()`.
3. **Batch Stacking**:
   - Stacked into a tensor batch of shape `[2, 4]` (Batch Size: 2, Sequence Length: 4).

---

## 4. Forward Pass Execution & Logits Output

When feeding the 2D batch tensor into `DummyGPTModel(cfg=GPT_CONFIG_124M)`:

1. **Embedding Calculation**:
   - `tok_embeds` has shape `[2, 4, 768]`.
   - `pos_embeds` has shape `[4, 768]`.
   - Summed combined embeddings `x` has shape `[2, 4, 768]`.
2. **Pass Through Transformer Layers**:
   - Sequentially processed through 12 `DummyTransformerBlock` layers and `DummyLayerNorm`.
3. **Logits Projection**:
   - `out_head` projects shape `[2, 4, 768]` $\rightarrow$ `[2, 4, 50257]`.

### Final Output Tensor Shape:
$$\text{Output Shape} = [B, T, V] = [2, 4, 50257]$$
- **Batch Size ($B$)**: 2
- **Sequence Length ($T$)**: 4
- **Vocabulary Size ($V$)**: 50,257

---

## 5. Layer Normalization Implementation (`LayerNorm`)

Layer Normalization stabilizes deep neural network training by normalizing activation outputs across features/channels for each sample independently.

### Custom `LayerNorm` Module (`nn.Module`)

The custom implementation defines learnable scale ($\gamma$) and shift ($\beta$) parameters along with a small epsilon ($\epsilon$) for numerical stability:

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # Learnable scale (gamma)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Learnable shift (beta)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

### Key Steps & Formulas:

1. **Mean & Variance Calculation**:
   - Computes mean $\mu$ and biased variance $\sigma^2$ (`unbiased=False`) across the last dimension (`dim=-1`):
     $$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$

2. **Standardization**:
   - Normalizes input features to zero mean and unit variance:
     $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

3. **Scale & Shift Transformation**:
   - Multiplies by `scale` ($\gamma$) and adds `shift` ($\beta$):
     $$y = \gamma \odot \hat{x} + \beta$$

4. **Empirical Verification**:
   - After passing a sample output `op` through `ln = LayerNorm(6)`, verifying `norm_op.mean(dim=-1, keepdim=True)` yields $\approx 0$ (e.g. `[0.0000, -0.0000]`).
   - Verifying `norm_op.var(dim=-1, keepdim=True, unbiased=False)` yields $\approx 1.0$ (e.g. `[0.9998, 0.9999]`).

---

## 6. Feed-Forward Network & GELU Activation (`FeedForward`, `GELU`)

The Transformer block utilizes a position-wise Feed-Forward Network (FFN) with GELU activation to process features after multi-head attention.

### 1. GELU Activation (`GELU` Module)

Gaussian Error Linear Unit (GELU) provides a smooth, non-linear activation function defined using the cumulative distribution function of the Gaussian distribution:

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))
```

### 2. Position-wise Feed-Forward Network (`FeedForward` Module)

The `FeedForward` module expands hidden embeddings by a factor of 4 (`4 * emb_dim`) using an internal linear expansion layer, applies GELU non-linearity, and projects back to `emb_dim`:

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)
```

- **Input Shape**: `[Batch Size, Sequence Length, emb_dim]` (e.g., `[2, 3, 768]`)
- **Intermediate Projection**: `[Batch Size, Sequence Length, 4 * emb_dim]` (e.g., `[2, 3, 3072]`)
- **Output Shape**: `[Batch Size, Sequence Length, emb_dim]` (e.g., `[2, 3, 768]`)


