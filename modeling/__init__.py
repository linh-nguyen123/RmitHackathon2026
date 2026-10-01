"""
Modeling package for 5-Fold Stratified Cross Validation,
Transformer backbones (mDeBERTa, PhoBERT), ROC-AUC metric, and Offline Kaggle Ensembling.
"""

from .config import ModelConfig, TrainingConfig
from .dataset import TextDataset, create_stratified_folds
from .model import TransformerClassifier, get_model_and_tokenizer
from .metrics import compute_roc_auc, compute_eval_metrics
from .train import KFoldTrainer
from .infer import EnsemblePredictor

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "TextDataset",
    "create_stratified_folds",
    "TransformerClassifier",
    "get_model_and_tokenizer",
    "compute_roc_auc",
    "compute_eval_metrics",
    "KFoldTrainer",
    "EnsemblePredictor",
]
