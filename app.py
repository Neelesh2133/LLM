"""
GPT-2 Interactive UI — FastAPI Backend
Serves pretrained GPT-2 for text generation and fine-tuned classifier for spam detection.
"""

import os
import sys
import torch
import torch.nn as nn
import tiktoken
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Add project dirs to path so we can import the model definitions
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "05_finetuning"))
sys.path.insert(0, str(PROJECT_ROOT / "03_gpt_architecture"))

from models import (  # noqa: E402
    GPTModel,
    text_to_token_ids,
    token_ids_to_text,
)

# ---------------------------------------------------------------------------
# GPT-2 124M Configurations
# The pretrained model was trained with context_length=256 and no QKV bias.
# The classifier was loaded from OpenAI GPT-2 weights (1024 ctx, QKV bias).
# ---------------------------------------------------------------------------
GPT_CONFIG_PRETRAINED = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

GPT_CONFIG_CLASSIFIER = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = tiktoken.get_encoding("gpt2")

# ---------------------------------------------------------------------------
# Model holders (loaded on startup)
# ---------------------------------------------------------------------------
gen_model = None
cls_model = None
models_loaded = {"generation": False, "classification": False}


def load_generation_model():
    """Load the pretrained GPT-2 model for text generation."""
    global gen_model
    model_path = PROJECT_ROOT / "04_pretraining" / "model.pth"
    if not model_path.exists():
        print(f"[WARNING] Generation model not found at {model_path}")
        return

    print(f"[INFO] Loading generation model from {model_path}...")
    model = GPTModel(GPT_CONFIG_PRETRAINED)
    state_dict = torch.load(str(model_path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    gen_model = model
    models_loaded["generation"] = True
    print(f"[INFO] Generation model loaded on {DEVICE}")


def load_classification_model():
    """Load the fine-tuned GPT-2 spam classifier."""
    global cls_model
    model_path = PROJECT_ROOT / "05_finetuning" / "text_classifier.pth"
    if not model_path.exists():
        print(f"[WARNING] Classification model not found at {model_path}")
        return

    print(f"[INFO] Loading classification model from {model_path}...")
    model = GPTModel(GPT_CONFIG_CLASSIFIER)
    # Replace the output head with a 2-class linear layer (matching saved weights)
    model.out_head = nn.Linear(GPT_CONFIG_CLASSIFIER["emb_dim"], 2, bias=True)
    state_dict = torch.load(str(model_path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    cls_model = model
    models_loaded["classification"] = True
    print(f"[INFO] Classification model loaded on {DEVICE}")


# ---------------------------------------------------------------------------
# Generation with temperature + top-k sampling
# ---------------------------------------------------------------------------
def generate(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Generate text continuation with temperature-scaled top-k sampling."""
    if gen_model is None:
        return "[Error: Generation model not loaded]"

    input_ids = text_to_token_ids(prompt, TOKENIZER).to(DEVICE)
    gen_model.eval()

    for _ in range(max_new_tokens):
        # Crop to context window
        idx_cond = input_ids[:, -GPT_CONFIG_PRETRAINED["context_length"]:]

        with torch.no_grad():
            logits = gen_model(idx_cond)

        # Focus on last token
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_clamped = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, top_k_clamped)
                min_top_value = top_values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_top_value,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy (temperature=0)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop at <|endoftext|>
        if idx_next.item() == TOKENIZER.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]:
            break

        input_ids = torch.cat((input_ids, idx_next), dim=1)

    # Decode only the generated portion
    output_ids = input_ids[:, len(text_to_token_ids(prompt, TOKENIZER)[0]):]
    return token_ids_to_text(output_ids, TOKENIZER)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
CLASSIFIER_MAX_LENGTH = 120
PAD_TOKEN_ID = 50256  # <|endoftext|>


def classify(text: str) -> dict:
    """Classify text as spam or ham using the fine-tuned model."""
    if cls_model is None:
        return {"label": "unknown", "confidence": 0.0, "logits": [0, 0], "error": "Classification model not loaded"}

    cls_model.eval()

    # Tokenize and pad/truncate
    input_ids = TOKENIZER.encode(text)
    input_ids = input_ids[:CLASSIFIER_MAX_LENGTH]
    input_ids += [PAD_TOKEN_ID] * (CLASSIFIER_MAX_LENGTH - len(input_ids))

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits = cls_model(input_tensor)[:, -1, :]  # (1, num_classes)

    probs = torch.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1).item()
    confidence = probs[0, pred].item()

    return {
        "label": "spam" if pred == 1 else "ham",
        "confidence": round(confidence, 4),
        "logits": [round(logits[0, 0].item(), 4), round(logits[0, 1].item(), 4)],
    }


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(title="GPT-2 Interactive UI", version="1.0.0")


# Request / Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=50, ge=1, le=200)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.on_event("startup")
async def startup():
    """Load both models when the server starts."""
    load_generation_model()
    load_classification_model()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": models_loaded,
    }


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    generated_text = generate(
        prompt=req.prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    return {
        "prompt": req.prompt,
        "generated": generated_text,
        "full_text": req.prompt + generated_text,
    }


@app.post("/api/classify")
async def api_classify(req: ClassifyRequest):
    result = classify(req.text)
    return result


# Serve static files
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse(str(static_dir / "index.html"))
