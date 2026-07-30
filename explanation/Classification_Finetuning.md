# Classification Finetuning Data Preparation (`Classification_Finetuning.ipynb`)

This document outlines the data preparation and downloading workflow used for **Classification Finetuning** in [`code/Classification_Finetuning.ipynb`](file:///d:/projects/LLM/code/Classification_Finetuning.ipynb).

---

## 📌 Environment Setup

The notebook checks dependencies required for data processing, tokenization, model loading, and fine-tuning:

```python
from importlib.metadata import version
pkgs = ["matplotlib", "numpy", "tiktoken", "torch", "tensorflow", "pandas"]
for p in pkgs:
    print(f"{p} version: {version(p)}")
```

---

## 📥 Dataset Download & Extraction

The **SMS Spam Collection** dataset is downloaded from UCI machine learning repository (or fallback mirror), unzipped, and stored as TSV:

```python
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
```

---

## 📊 Dataset Loading & Class Distribution

The raw dataset is loaded using Pandas into a structured DataFrame containing `Label` and `Text` columns:

```python
import pandas as pd

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
df.head(5)
```

### Class Distribution (Imbalance)
- **`ham`** (legitimate messages): **4,825**
- **`spam`** (unwanted messages): **747**

> ⚠️ **Note:** The dataset displays class imbalance (~86.5% ham vs ~13.5% spam), which must be handled during dataset splitting and performance evaluation metrics (e.g., tracking F1-score/Accuracy alongside loss).
