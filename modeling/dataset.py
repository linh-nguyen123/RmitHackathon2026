"""
Dataset and Cross-Validation Data Splitters for NLP Modeling.
Handles PyTorch tensors, tokenization collators, and Stratified K-Fold splits.
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:  # type: ignore
        pass

from sklearn.model_selection import StratifiedKFold


class TextDataset(Dataset):
    """
    PyTorch Dataset wrapping raw or preprocessed text and optional target labels.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[Union[List[int], np.ndarray]] = None,
        tokenizer: Any = None,
        max_length: int = 256,
    ):
        self.texts = list(texts)
        self.labels = labels if labels is None else np.array(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = str(self.texts[idx])

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt" if TORCH_AVAILABLE else None,
            )
            item = {
                "input_ids": encoding["input_ids"].squeeze(0) if TORCH_AVAILABLE else encoding["input_ids"],
                "attention_mask": encoding["attention_mask"].squeeze(0) if TORCH_AVAILABLE else encoding["attention_mask"],
            }
            if "token_type_ids" in encoding:
                item["token_type_ids"] = encoding["token_type_ids"].squeeze(0) if TORCH_AVAILABLE else encoding["token_type_ids"]
        else:
            item = {"text": text}

        if self.labels is not None:
            label_val = self.labels[idx]
            if TORCH_AVAILABLE:
                item["labels"] = torch.tensor(label_val, dtype=torch.long)
            else:
                item["labels"] = label_val

        return item


def create_stratified_folds(
    df: Any,
    target_col: str,
    n_splits: int = 5,
    seed: int = 42,
) -> Any:
    """
    Generates balanced folds using StratifiedKFold and assigns a 'fold' column (0 to n_splits-1).
    Works with pandas DataFrames or polars DataFrames.
    """
    # Check if pandas or polars
    df_copy = df.copy()

    # If pandas DataFrame
    if hasattr(df_copy, "to_numpy"):
        y = df_copy[target_col].to_numpy()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        df_copy["fold"] = -1
        for fold, (_, val_idx) in enumerate(skf.split(df_copy, y)):
            df_copy.loc[val_idx, "fold"] = fold
        return df_copy
    else:
        # Fallback for dict / arrays
        y = np.array(df_copy[target_col])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = np.zeros(len(y), dtype=int)
        for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            folds[val_idx] = fold
        df_copy["fold"] = folds
        return df_copy
