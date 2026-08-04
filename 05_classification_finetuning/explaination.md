# Classification Finetuning Data Preparation (`Classification_Finetuning.ipynb`)

This document outlines the data preparation and downloading workflow used for **Classification Finetuning** in [`05_classification_finetuning/Classification_Finetuning.ipynb`](file:///d:/projects/LLM/05_classification_finetuning/Classification_Finetuning.ipynb).



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

## ⚖️ Creating a Balanced Dataset

To prevent bias toward the majority class (`ham`), a subset of `ham` samples is randomly sampled to match the total count of `spam` instances (747 each):

```python
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

balanced_df = create_balanced_dataset(df)
# Output: ham: 747, spam: 747

# Map labels: ham -> 0, spam -> 1
map_dict = {"ham": 0, "spam": 1}
balanced_df["Label"] = balanced_df["Label"].map(map_dict)
```

---

## 🔀 Dataset Splitting Strategy

The balanced dataset is split into training, validation, and test sets using a reproducible random shuffle:

```python
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df
```

