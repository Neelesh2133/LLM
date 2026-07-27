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

## 5. Layer Normalization Fundamentals

Layer Normalization stabilizes deep neural network training by normalizing activation outputs across features/channels for each sample independently.

### Goal: $\text{Mean} = 0, \text{Variance} = 1$

1. **Calculating Feature Mean & Variance**:
   - Given an activation tensor `op` (e.g. from a Linear layer with ReLU):
     $$\mu = \text{op.mean}(\text{dim}=-1, \text{keepdim}=\text{True})$$
     $$\sigma^2 = \text{op.var}(\text{dim}=-1, \text{keepdim}=\text{True})$$

2. **Applying Normalization Formula**:
   $$\text{norm} = \frac{\text{op} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

3. **Verification**:
   - Calculating `norm.var(dim=1, keepdim=True)` yields `tensor([[1.0000], [1.0000]])`, confirming that the normalized outputs achieve a unit variance of 1 (and zero mean).
