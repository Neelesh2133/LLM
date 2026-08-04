# 📊 Step 06 — Instruction Finetuning: Visual Workflow

> **Companion visualization for** [`Instruction_Finetuning.md`](file:///d:/projects/LLM/06_instruction_finetuning/Instruction_Finetuning.md)

---

## 1. Instruction Prompt Formatting & Masking Pipeline

```
+-----------------------------------------------------------------------+
| Prompt Template:                                                      |
| Below is an instruction that describes a task. Write a response...   |
| ### Instruction: {instruction}                                       |
| ### Input: {input}                                                    |
| ### Response: {response}                                              |
+-----------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
| Masked Target Tokens (Ignore Loss on Prompt):                         |
| [-100, -100, ..., -100, token_1, token_2, token_3, <|endoftext|>]     |
+-----------------------------------------------------------------------+
```

---

## 2. Supervised Fine-Tuning Flow

```
Formatted Dataset ---> GPT Model ---> Token Predictions ---> Cross-Entropy (Response Only) ---> Weight Update
```
