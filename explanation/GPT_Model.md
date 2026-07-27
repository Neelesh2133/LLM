# GPT Model Architecture Explained

Based on the `code/GPT_Model.ipynb` file, this document provides a detailed breakdown of the mock architecture for a GPT model, named `DummyGPTModel`, and how tokenization works for creating input batches.

## 1. `DummyGPTModel` Architecture
The core component is a PyTorch module `DummyGPTModel` which mirrors the macro-structure of a standard GPT architecture. It relies on a configuration dictionary (`cfg`) to set hyperparameters like vocabulary size (`vocab_size`), embedding dimension (`emb_dim`), context length (`context_length`), etc. 

### Key Components:
- **Embeddings (`tok_emb` & `pos_emb`)**: 
  - `tok_emb`: A standard `nn.Embedding` that converts input token indices (from the vocabulary) into dense vectors of size `emb_dim`.
  - `pos_emb`: A learned positional embedding layer. Since Transformers do not have a built-in sense of sequence order, this provides spatial context by embedding the positions (up to `context_length`) into vectors of size `emb_dim`.
- **Dropout (`drop_emb`)**: A regularization layer to randomly drop out components of the embeddings to prevent overfitting.
- **Transformer Blocks (`trf_blocks`)**: A sequence of `n_layers` Transformer blocks. Currently, these use a placeholder class `DummyTransformerBlock` which just passes the input through without modification.
- **Final Layer Normalization (`final_norm`)**: A placeholder `DummyLayerNorm` applied to the output of the transformer blocks to stabilize training.
- **Output Head (`out_head`)**: A linear layer (`nn.Linear` without bias) that maps the final hidden states back to the original vocabulary size to output the `logits` (raw prediction scores for the next token).

### The Forward Pass
The `forward(self, in_idx)` function defines the data flow:
1. It takes an input tensor of token indices (`in_idx`).
2. Calculates token and positional embeddings based on sequence length.
3. Adds the two embeddings together and applies dropout.
4. Passes the result through the sequence of Transformer blocks and final layer normalization.
5. Uses the output head to generate the prediction `logits`.

## 2. Tokenization and Batching Setup
The file also sets up data inputs using the `tiktoken` library:
- Loads the standard **GPT-2 tokenizer** encoding.
- Encodes sample text sequences (`"every effort moves you"` and `"every day holds a"`) into token IDs.
- Stacks these token arrays into a PyTorch batch tensor.
- This resulting 2D tensor matches the shape required by the `DummyGPTModel`'s `forward` method (as `in_idx`).
