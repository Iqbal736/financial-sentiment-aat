"""
Data loading and stratification pipeline for Financial PhraseBank.

Financial PhraseBank (Malo et al., 2014):
- 3 classes: Negative (0), Neutral (1), Positive (2)
- Stratified Split: 70% Train, 15% Validation, 15% Test
"""

import os
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

LABEL_MAP: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

SUBSET_FILE_MAP: Dict[str, str] = {
    "sentences_50agree": "Sentences_50Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}


def load_from_hf_hub_direct(subset: str = "sentences_75agree") -> pd.DataFrame:
    """
    Directly download and extract Financial PhraseBank from Hugging Face hub repository.
    This provides robust loading even when HF `datasets` disables dataset execution scripts.
    """
    from huggingface_hub import hf_hub_download

    print(f"Fetching Financial PhraseBank ({subset}) directly from Hugging Face hub...")
    zip_path = hf_hub_download(
        repo_id="financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )

    filename_in_zip = SUBSET_FILE_MAP.get(subset.lower(), "Sentences_75Agree.txt")
    target_inner_path = f"FinancialPhraseBank-v1.0/{filename_in_zip}"

    rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(target_inner_path) as f:
            for line in f.read().decode("latin-1").splitlines():
                line = line.strip()
                if not line or "@" not in line:
                    continue
                sentence, label_str = line.rsplit("@", 1)
                sentence = sentence.strip()
                label_str = label_str.strip().lower()
                if label_str in LABEL_MAP:
                    rows.append({
                        "sentence": sentence,
                        "label": LABEL_MAP[label_str],
                        "label_text": label_str,
                    })

    df = pd.DataFrame(rows)
    print(f"Successfully extracted {len(df)} sentences for {subset}.")
    return df


def load_financial_phrasebank(subset: str = "sentences_75agree") -> pd.DataFrame:
    """
    Load Financial PhraseBank dataset, trying datasets.load_dataset first,
    falling back to direct zip extraction if HF script errors occur.
    """
    try:
        from datasets import load_dataset
        raw = load_dataset("financial_phrasebank", subset)
        df = pd.DataFrame(raw["train"])
        if "label" in df.columns and "sentence" in df.columns:
            # Add label text column for transparency
            inv_map = {0: "negative", 1: "neutral", 2: "positive"}
            df["label_text"] = df["label"].map(inv_map)
            print(f"Loaded {len(df)} rows via datasets library.")
            return df
    except Exception as e:
        print(f"datasets.load_dataset did not succeed ({e}). Falling back to direct hub download.")

    return load_from_hf_hub_direct(subset=subset)


def prepare_financial_dataset(
    config_path: str = "config/config.yaml",
    subset: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads dataset, splits into stratified train (70%), val (15%), test (15%),
    and writes raw, train, val, test CSVs into the data directory.
    """
    # Load configuration
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    subset = subset or cfg.get("data", {}).get("subset", "sentences_75agree")
    seed = seed if seed is not None else cfg.get("project", {}).get("seed", 42)

    raw_path = Path(cfg.get("data", {}).get("raw_path", "data/raw_phrasebank.csv"))
    train_path = Path(cfg.get("data", {}).get("train_path", "data/train.csv"))
    val_path = Path(cfg.get("data", {}).get("val_path", "data/val.csv"))
    test_path = Path(cfg.get("data", {}).get("test_path", "data/test.csv"))

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_financial_phrasebank(subset=subset)
    df.to_csv(raw_path, index=False)
    print(f"Saved full raw dataset to {raw_path}")

    # Stratified Train (70%), Temp (30%)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    # Stratified Val (15%), Test (15%) from Temp (50% each)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nDataset Split Summary:")
    print(f"  Train : {len(train_df):5d} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(val_df):5d} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test_df):5d} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Total : {len(df):5d}")

    print("\nClass Distribution (Percentages):")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df["label_text"].value_counts(normalize=True).to_dict()
        dist_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in dist.items())
        print(f"  {name:5s}: {dist_str}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_financial_dataset()

