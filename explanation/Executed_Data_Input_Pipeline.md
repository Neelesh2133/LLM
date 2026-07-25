# Executed Data Input Pipeline (`Executed_Data_Input_Pipeline.ipynb`)

This document records the executed runtime findings, data shapes, tensor outputs, and verification details for **`Executed_Data_Input_Pipeline.ipynb`**.

---

## 📌 Execution Summary

`Executed_Data_Input_Pipeline.ipynb` contains the full execution trace of the data pre-processing and embedding generation pipeline ran against `verdict.txt`.

---

## 1. Raw Text Statistics

- **Input File**: `verdict.txt` (Edith Wharton's short story *"The Verdict"*)
- **Total Raw Characters**: `20,637` characters
- **Regex Preprocessed Token Count**: `4,690` tokens

---

## 2. Vocabulary & Tokenizer Benchmark

### Custom Regex Tokenizer (`SimpleTokenizerV1`)
- **Unique Vocabulary Size**: `1,130` tokens (without special tokens)
- **With Special Tokens (`<|unk|>`, `<|endoftext|>)**: `1,132` tokens
- **Output Sample (`encode/decode`)**:
  - Sample Input Text: `"You are a painter , you paint good pride hello sambar"`
  - Encoded Token IDs: `[1126, 172, 38, 703, 11, 1130, 702, 461, 762, 1131, 1131]`
  - Decoded Text: `"You are a painter, <|unk|> paint good pride <|unk|> <|unk|>"`

### OpenAI `tiktoken` BPE Tokenizer (GPT-2)
- **Vocabulary Size**: `50,257`
- **Output Sample (`hello world`)**: Token IDs `[31373, 995]`
- **Sample Paragraph Token Count**: `83` BPE tokens

---

## 3. Sliding Window Sampling Outputs

Using `GPTDatasetV1` and `create_dataloader_v1` with configuration:
- `max_length = 4`
- `stride = 4`
- `batch_size = 8`

```python
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
inputs, targets = next(iter(dataloader))
```

### Shape Verification
- **Input Tensor Shape (`inputs`)**: `torch.Size([8, 4])`
- **Target Tensor Shape (`targets`)**: `torch.Size([8, 4])`

#### Batch 1 Sample Input Tokens:
```text
Inputs Shape: torch.Size([8, 4])
Targets Shape: torch.Size([8, 4])
```

---

## 4. Embedding Layer Output Verification

### Token Embeddings
- `vocab_size = 50257`
- `output_dim = 256`
- `token_embedding_layer = torch.nn.Embedding(50257, 256)`
- `token_embeddings = token_embedding_layer(inputs)`
- **Output Shape**: `torch.Size([8, 4, 256])`

### Positional Embeddings
- `context_len = 4`
- `pos_emb_layer = torch.nn.Embedding(4, 256)`
- `pos_emb = pos_emb_layer(torch.arange(4))`
- **Output Shape**: `torch.Size([4, 256])`

### Combined Embeddings Matrix
- `input_emb = token_embeddings + pos_emb`
- **Final Output Shape**: `torch.Size([8, 4, 256])`

---

## 💡 Key Takeaway & Verification Matrix

| Component | Class / Layer | Input Shape | Output Shape |
|---|---|---|---|
| **Tokenizer** | `tiktoken.get_encoding("gpt2")` | Raw string | 1D list of integer token IDs |
| **Dataset Windowing** | `GPTDatasetV1` | Full token ID list | Tensor pairs `(input_ids, target_ids)` |
| **DataLoader** | `DataLoader` | Dataset | `[8, 4]` (batch of sequence chunks) |
| **Token Embedding** | `nn.Embedding(50257, 256)` | `[8, 4]` | `[8, 4, 256]` |
| **Positional Embedding**| `nn.Embedding(4, 256)` | `[4]` | `[4, 256]` (broadcasted) |
| **Combined Input** | `token_embeddings + pos_emb` | `[8, 4, 256]` + `[4, 256]` | `[8, 4, 256]` |
