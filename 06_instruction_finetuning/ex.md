# 🎯 Step 06 — Instruction Finetuning

Welcome to the **Instruction Finetuning** module!

Instruction fine-tuning adapts a pretrained Large Language Model (LLM) to follow human instructions, act as an AI assistant, and format responses effectively (e.g., Alpaca-style instruction dataset formats, prompt formatting, masked target cross-entropy loss, and evaluation).

---

## 📂 Module Contents

- `Instruction_Finetuning.ipynb`: Jupyter notebook covering dataset loading, prompt template formatting, target masking, and training loop.
- `Instruction_Finetuning.md`: Detailed step-by-step breakdown of instruction dataset formatting and loss computation logic.
- `Visualization_Instruction_Finetuning.md`: Visual diagrams outlining prompt formatting, context masking, and fine-tuning pipelines.

---

## 💡 Key Concepts Covered
1. **Instruction Prompt Templates**: Structuring raw input/output pairs into standard prompt formats (Instruction, Input, Response).
2. **Loss Masking (Prompt Masking)**: Masking instruction prompt tokens so loss is computed solely on response tokens.
3. **Supervised Fine-Tuning (SFT)**: Fine-tuning GPT weights end-to-end on formatted instruction-response pairs.
