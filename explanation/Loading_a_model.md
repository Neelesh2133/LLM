# Loading a Saved GPT Model (`Loading_a_model.ipynb`)

This document provides a breakdown of **`code/Loading_a_model.ipynb`** ([notebook link](file:///d:/projects/LLM/code/Loading_a_model.ipynb)), covering how to initialize and load trained model state dict parameters into a `GPTModel` instance using PyTorch.

---

## 📌 Model Configuration & Setup

Loading a pre-trained or fine-tuned model requires initializing the model architecture with matching structural parameters (`GPT_CONFIG_124M`) before loading checkpoint weights (`state_dict`).

```python
import torch
from gpt_model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size (BPE tokens)
    "context_length": 256,    # Context length window
    "emb_dim": 768,           # Hidden embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Transformer layers count
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias flag
}

# 1. Instantiate model structure
model = GPTModel(GPT_CONFIG_124M)

# 2. Select compute device and load model weights checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device))

# 3. Set evaluation mode for inference (disables dropout)
model.eval()
```

---

## 🔑 Key Steps for Checkpoint Loading

1. **Architecture Matching**: The configuration dictionary (`cfg`) supplied to `GPTModel` must match the exact dimensions (`vocab_size`, `emb_dim`, `n_layers`, `n_heads`) used during training.
2. **Device-Aware Loading**: Using `map_location=device` inside `torch.load()` ensures parameters are appropriately dispatched to GPU (`cuda`) or CPU without device mismatch issues.
3. **`state_dict` Verification**: Successful loading returns `<All keys matched successfully>`, verifying all layer weights align with the model architecture.
4. **Pre-trained OpenAI Weights Integration**: Pre-trained weights from OpenAI models (e.g., GPT-2 124M) can be fetched and extracted into Python parameter dictionaries using `download_and_load_gpt2`.

---

## 📥 Downloading Pre-trained OpenAI GPT-2 Weights

You can fetch official pre-trained GPT-2 checkpoints directly from OpenAI's repository using the helper utility `download_and_load_gpt2`:

```python
from gpt_code import download_and_load_gpt2

# Download and load GPT-2 124M weights into memory
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

print("Settings:", settings)
print("Token embedding weight tensor shape:", params["wte"].shape) # Output: (50257, 768)
```

- **`settings`**: Contains hyperparameter metadata such as `n_vocab` (50257), `n_ctx` (1024), `n_embd` (768), `n_head` (12), `n_layer` (12).
- **`params`**: Dictionary containing NumPy weight arrays for embeddings (`wte`, `wpe`), transformer blocks, and layer norms.


