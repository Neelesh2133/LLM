# Building an LLM from Scratch 🚀

A step-by-step, hands-on implementation of the core building blocks of a Large Language Model (LLM) from the ground up using PyTorch. This repository is organized into distinct **topic-based subfolders**, pairing each implementation notebook directly with its corresponding detailed explanation guide.

---

## 📂 Repository Structure

```text
LLM/
├── 📁 01_data_pipeline/                # Text reading, tokenization, dataset & embeddings
│   ├── Data_Input_Pipeline.ipynb
│   ├── Data_Input_Pipeline.md
│   ├── 📊 Visualization_Data_Pipeline.md
│   ├── Executed_Data_Input_Pipeline.md
│   └── verdict.txt
├── 📁 02_attention_mechanism/         # Self-attention, multi-head attention & matrices
│   ├── Attention_Mechanism.ipynb
│   ├── Attention_Mechanism.md
│   ├── 📊 Visualization_Attention.md
│   └── Multiheadattention.py
├── 📁 03_gpt_architecture/            # Transformer block & full GPT model construction
│   ├── GPT_Model.ipynb
│   ├── GPT_Model.md
│   ├── Transformer_Block.md
│   ├── 📊 Visualization_GPT_Architecture.md
│   └── gpt_model.py
├── 📁 04_pretraining/                 # Pre-training loop, evaluation loss & weight loading
│   ├── Pretraining_Model.ipynb
│   ├── Pretraining_Model.md
│   ├── Loading_a_model.ipynb
│   ├── Loading_a_model.md
│   ├── 📊 Visualization_Pretraining.md
│   └── gpt_code.py
├── 📁 05_classification_finetuning/   # Classification fine-tuning & dataset balancing
│   ├── Classification_Finetuning.ipynb
│   ├── Classification_Finetuning.md
│   └── 📊 Visualization_Finetuning.md
├── 📁 06_instruction_finetuning/      # Instruction fine-tuning & dataset formatting
│   ├── Instruction_Finetuning.md
│   └── 📊 Visualization_Instruction_Finetuning.md
├── 📁 ui/                            # Interactive Web UI & FastAPI backend
│   ├── app.py
│   ├── index.html
│   ├── style.css
│   └── app.js
├── README.md                           # Project Overview & Setup Instructions
└── requirements.txt                    # Project Dependencies
```

---

## 🗂️ Modules & Detailed Explanations

| Module | 💻 Implementation Code | 📘 Explanation Guide | 📊 Visual Workflow | Description |
|---|---|---|---|---|
| **01. Data Input Pipeline** | [Data_Input_Pipeline.ipynb](file:///d:/projects/LLM/01_data_pipeline/Data_Input_Pipeline.ipynb) | [Data_Input_Pipeline.md](file:///d:/projects/LLM/01_data_pipeline/Data_Input_Pipeline.md) | [Visualization_Data_Pipeline.md](file:///d:/projects/LLM/01_data_pipeline/Visualization_Data_Pipeline.md) | Text reading, Regex/NLTK/BPE tokenization, sliding window dataset, and token/positional embeddings. |
| **02. Attention Mechanism** | [Attention_Mechanism.ipynb](file:///d:/projects/LLM/02_attention_mechanism/Attention_Mechanism.ipynb) | [Attention_Mechanism.md](file:///d:/projects/LLM/02_attention_mechanism/Attention_Mechanism.md) | [Visualization_Attention.md](file:///d:/projects/LLM/02_attention_mechanism/Visualization_Attention.md) | Dot-product self-attention, matrix multiplication, trainable QKV weights, scaling factor, and PyTorch modules. |
| **03. GPT Model Architecture** | [GPT_Model.ipynb](file:///d:/projects/LLM/03_gpt_architecture/GPT_Model.ipynb) | [GPT_Model.md](file:///d:/projects/LLM/03_gpt_architecture/GPT_Model.md) | [Visualization_GPT_Architecture.md](file:///d:/projects/LLM/03_gpt_architecture/Visualization_GPT_Architecture.md) | Full GPT model class (`GPTModel`), Layer Normalization, multi-head attention blocks, and forward pass output shapes. |
| **04. Model Pretraining** | [Pretraining_Model.ipynb](file:///d:/projects/LLM/04_pretraining/Pretraining_Model.ipynb) | [Pretraining_Model.md](file:///d:/projects/LLM/04_pretraining/Pretraining_Model.md) | [Visualization_Pretraining.md](file:///d:/projects/LLM/04_pretraining/Visualization_Pretraining.md) | Pretraining GPT on unlabeled text, cross-entropy loss, perplexity calculation, training loop, loss plotting, and loading saved weights. |
| **05. Classification Finetuning** | [Classification_Finetuning.ipynb](file:///d:/projects/LLM/05_classification_finetuning/Classification_Finetuning.ipynb) | [Classification_Finetuning.md](file:///d:/projects/LLM/05_classification_finetuning/Classification_Finetuning.md) | [Visualization_Finetuning.md](file:///d:/projects/LLM/05_classification_finetuning/Visualization_Finetuning.md) | Fine-tuning GPT for text classification, SMS spam collection dataset downloading, class balancing, and split creation. |
| **06. Instruction Finetuning** | `Instruction_Finetuning.ipynb` | [Instruction_Finetuning.md](file:///d:/projects/LLM/06_instruction_finetuning/Instruction_Finetuning.md) | [Visualization_Instruction_Finetuning.md](file:///d:/projects/LLM/06_instruction_finetuning/Visualization_Instruction_Finetuning.md) | Fine-tuning GPT to follow human instructions, prompt formatting, target token loss masking, and response generation evaluation. |

---

## 🛠️ Pipeline Overview

### 1. Data Input Pipeline ([01_data_pipeline/](file:///d:/projects/LLM/01_data_pipeline/))
Before feeding text into an LLM, raw text must be parsed and converted into embeddings.
- **Tokenization**: Exploring basic split-by-regex methods, NLTK word tokenizers, and Byte Pair Encoding (BPE) using OpenAI's `tiktoken` (GPT-2 vocabulary).
- **Special Tokens**: Handling unknown words (`<|unk|>`) and document boundary markers (`<|endoftext|>`).
- **Data Sampling**: Structuring text into inputs ($x$) and targets ($y$) using a custom PyTorch `Dataset` (`GPTDatasetV1`) and a sliding window dataloader.
- **Embeddings**: Creating trainable token embedding layers (`torch.nn.Embedding`) and positional embedding layers.

### 2. Attention Mechanisms ([02_attention_mechanism/](file:///d:/projects/LLM/02_attention_mechanism/))
Attention allows models to dynamically focus on relevant tokens across sequences.
- **Simple Self-Attention**: Calculating attention weights using dot-product similarity between raw inputs.
- **Matrix Operations**: Vectorizing the attention mechanism using PyTorch tensor multiplication.
- **Trainable QKV Weights**: Transitioning to trainable Query ($W_q$), Key ($W_k$), and Value ($W_v$) weight projections.
- **PyTorch Modules**: Modular encapsulation into `SelfAttention_v1` (`nn.Parameter`) and `SelfAttention_v2` (`nn.Linear`).

### 3. GPT Model Architecture ([03_gpt_architecture/](file:///d:/projects/LLM/03_gpt_architecture/))
- **Transformer Block**: Assembling multi-head self-attention, layer normalization, GELU activation, shortcut connections, and feed-forward networks.
- **GPT Architecture**: Stacking $N$ transformer blocks with positional embeddings and output projection head.

### 4. Model Pretraining ([04_pretraining/](file:///d:/projects/LLM/04_pretraining/))
- **Cross-Entropy & Perplexity**: Evaluating model cross-entropy loss and token perplexity.
- **Training Pipeline**: Batch loss computation (`calc_loss_batch`), data loader iteration (`calc_loss_loader`), and AdamW optimization loop (`train_model_simple`).
- **Loss Plotting & Sampling**: Dual-axis plotting of training vs validation loss across epochs and tokens seen, alongside text sampling generation.

### 5. Classification Finetuning ([05_classification_finetuning/](file:///d:/projects/LLM/05_classification_finetuning/))
- **Dataset Preparation**: Downloading and extracting the SMS Spam Collection dataset.
- **Class Balancing**: Undersampling majority `ham` class to match `spam` instances (747 samples each).
- **Dataset Splitting**: Train, validation, and test dataset creation using randomized shuffling.

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA support if available)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Neelesh2133/LLM.git
   cd LLM
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
