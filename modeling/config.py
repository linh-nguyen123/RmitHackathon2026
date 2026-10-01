"""
Configuration Dataclasses for Model and Training Pipeline.
Supports Kaggle offline configurations and multi-fold tracking.
"""

from dataclasses import dataclass, field, asdict
import json
import os
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """
    Configuration parameters for Hugging Face transformer backbone.
    """
    # Hugging Face hub ID or local folder path (for offline Kaggle)
    model_name_or_path: str = "microsoft/mdeberta-v3-base"
    # 'mdeberta' or 'phobert'
    model_type: str = "mdeberta"
    # Number of target classes: 2 for binary classification
    num_labels: int = 2
    # Maximum token sequence length
    max_length: int = 256
    # Dropout rate for classification head
    dropout_rate: float = 0.2
    # Pooling strategy: True for mean pooling over non-masked tokens, False for CLS token
    use_mean_pooling: bool = True
    # Multi-sample dropout with varying dropout probabilities
    multi_sample_dropout: bool = True
    # Custom hidden layer projection size (None to keep transformer hidden_size)
    custom_head_dim: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """
    Configuration parameters for Stratified K-Fold cross validation,
    optimization hyperparameters, and checkpointing.
    """
    n_splits: int = 5
    seed: int = 42
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    epochs: int = 4
    warmup_ratio: float = 0.1
    # Mixed precision training with torch.cuda.amp
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    # Stop early if validation ROC-AUC does not improve
    early_stopping_patience: int = 2
    output_dir: str = "./checkpoints"
    # Set True when submitting on Kaggle without internet
    offline_mode: bool = False
    # Local directory where weights are mounted in Kaggle (e.g. /kaggle/input/my-models/)
    local_weights_dir: Optional[str] = None
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
