"""
Offline Inference & 5-Fold Ensemble Predictor for Kaggle Submissions.
Averages predictions across all fold checkpoints without internet access.
"""

import os
import glob
from typing import List, Optional, Any
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import ModelConfig
from .dataset import TextDataset
from .model import TransformerClassifier


class EnsemblePredictor:
    """
    Loads all trained fold checkpoints from a local Kaggle dataset directory,
    runs offline batch inference, and produces ensembled test predictions.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        model_config: ModelConfig,
        device: Optional[str] = None,
    ):
        self.checkpoints_dir = checkpoints_dir
        self.model_cfg = model_config

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Hugging Face Transformers are required for EnsemblePredictor.")

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Find all fold checkpoint files
        self.checkpoint_paths = sorted(
            glob.glob(os.path.join(checkpoints_dir, "model_fold_*.pt"))
        )
        if not self.checkpoint_paths:
            raise FileNotFoundError(
                f"No fold checkpoints found matching 'model_fold_*.pt' in {checkpoints_dir}"
            )

        print(f"Found {len(self.checkpoint_paths)} fold checkpoints in {checkpoints_dir}")

        # Load Tokenizer locally
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoints_dir,
            local_files_only=True,
            use_fast=True if model_config.model_type != "phobert" else False,
        )

    def _predict_with_single_model(
        self, checkpoint_path: str, dataloader: Any
    ) -> np.ndarray:
        """Loads a single fold model and returns predicted probabilities."""
        model = TransformerClassifier(
            self.model_cfg,
            pretrained_model_or_path=self.checkpoints_dir,
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        probs = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs["logits"]

                if self.model_cfg.num_labels == 1:
                    batch_probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                elif self.model_cfg.num_labels == 2:
                    batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                else:
                    batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()

                probs.extend(batch_probs)

        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.array(probs)

    def predict_proba(
        self,
        texts: List[str],
        batch_size: int = 32,
        voting_method: str = "mean",
    ) -> np.ndarray:
        """
        Ensemble prediction across all fold checkpoints.

        Args:
            texts: List of input texts.
            batch_size: Batch size for inference dataloader.
            voting_method: 'mean' (arithmetic), 'gmean' (geometric), or 'rank'.
        """
        dataset = TextDataset(texts, tokenizer=self.tokenizer, max_length=self.model_cfg.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        fold_predictions = []
        for i, ckpt_path in enumerate(self.checkpoint_paths):
            print(f"Predicting with checkpoint {i + 1}/{len(self.checkpoint_paths)}: {os.path.basename(ckpt_path)}")
            preds = self._predict_with_single_model(ckpt_path, dataloader)
            fold_predictions.append(preds)

        fold_preds_arr = np.array(fold_predictions)  # Shape: [num_folds, N, ...]

        if voting_method == "mean":
            ensemble_probs = np.mean(fold_preds_arr, axis=0)
        elif voting_method == "gmean":
            # Geometric mean: exp(mean(log(p + 1e-12)))
            log_preds = np.log(np.clip(fold_preds_arr, 1e-12, 1.0))
            ensemble_probs = np.exp(np.mean(log_preds, axis=0))
        elif voting_method == "rank":
            # Rank average
            from scipy.stats import rankdata
            ranked = np.array([rankdata(p) / len(p) for p in fold_preds_arr])
            ensemble_probs = np.mean(ranked, axis=0)
        else:
            ensemble_probs = np.mean(fold_preds_arr, axis=0)

        return ensemble_probs

    def generate_submission(
        self,
        test_df: Any,
        text_col: str,
        id_col: str,
        output_path: str = "submission.csv",
        proba_col: str = "prediction",
        batch_size: int = 32,
        voting_method: str = "mean",
    ):
        """
        Runs full pipeline and saves submission.csv ready for Kaggle scoring.
        """
        texts = test_df[text_col].tolist()
        probabilities = self.predict_proba(texts, batch_size=batch_size, voting_method=voting_method)

        test_df[proba_col] = probabilities
        sub = test_df[[id_col, proba_col]]
        sub.to_csv(output_path, index=False)
        print(f"Successfully generated Kaggle submission file: {output_path} ({len(sub)} rows)")
        return sub
