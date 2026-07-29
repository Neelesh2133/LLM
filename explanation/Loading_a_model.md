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

# 2. Load model weights checkpoint
# model.load_state_dict(torch.load("model.pth"))

# 3. Set evaluation mode for inference (disables dropout)
model.eval()
```

---

## 🔑 Key Steps for Checkpoint Loading

1. **Architecture Matching**: The configuration dictionary (`cfg`) supplied to `GPTModel` must match the exact dimensions (`vocab_size`, `emb_dim`, `n_layers`, `n_heads`) used during training.
2. **`state_dict` Loading**: `model.load_state_dict(torch.load(model_path))` maps saved weight tensors to the corresponding model parameters.
3. **Evaluation Mode**: Always call `model.eval()` before running inference to disable stochastic dropout layers (`nn.Dropout`) and ensure deterministic outputs.
