"""
Stratified 5-Fold Cross Validation Trainer.
Optimized for ROC-AUC metric, Mixed Precision (fp16), and Out-Of-Fold (OOF) generation.
"""

import os
import gc
import numpy as np
from typing import Dict, Any, Tuple, Optional

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import ModelConfig, TrainingConfig
from .dataset import TextDataset, create_stratified_folds
from .model import get_model_and_tokenizer
from .metrics import compute_roc_auc, compute_eval_metrics

try:
    from security.evaluator import RedTeamingCallback
except (ImportError, ValueError):
    try:
        from ..security.evaluator import RedTeamingCallback
    except (ImportError, ValueError):
        RedTeamingCallback = None

try:
    from preprocessing.cleaner import clean_text
except (ImportError, ValueError):
    try:
        from ..preprocessing.cleaner import clean_text
    except (ImportError, ValueError):
        clean_text = None


class KFoldTrainer:
    """
    Manages 5-Fold Stratified Cross Validation training loop,
    checkpointing best weights per fold, and computing out-of-fold metrics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        red_team_callback: Optional[Any] = "auto",
    ):
        self.model_cfg = model_config
        self.train_cfg = training_config

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Hugging Face Transformers are required for KFoldTrainer.")

        # Determine device
        if self.train_cfg.device:
            self.device = torch.device(self.train_cfg.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seeds
        torch.manual_seed(self.train_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.train_cfg.seed)
        np.random.seed(self.train_cfg.seed)

        os.makedirs(self.train_cfg.output_dir, exist_ok=True)

        # Initialize automatic Red-Teaming Callback
        if red_team_callback == "auto" and RedTeamingCallback is not None:
            self.red_team_callback = RedTeamingCallback(
                output_dir=self.train_cfg.output_dir,
                clean_fn=clean_text,
            )
        else:
            self.red_team_callback = red_team_callback

    def _get_optimizer_and_scheduler(self, model, num_training_steps: int):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.train_cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.train_cfg.learning_rate)
        num_warmup_steps = int(num_training_steps * self.train_cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    def evaluate(self, model, dataloader) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Runs inference over a dataloader and computes validation metrics.
        Returns: (metrics_dict, predicted_probabilities)
        """
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                labels = batch.get("labels")
                if labels is not None:
                    all_labels.extend(labels.numpy() if hasattr(labels, "numpy") else labels)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                logits = outputs["logits"]

                if self.model_cfg.num_labels == 1:
                    probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                elif self.model_cfg.num_labels == 2:
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                all_probs.extend(probs)

        probs_arr = np.array(all_probs)
        if len(all_labels) > 0:
            labels_arr = np.array(all_labels)
            metrics = compute_eval_metrics(labels_arr, probs_arr)
        else:
            metrics = {}

        return metrics, probs_arr

    def train_fold(
        self,
        fold: int,
        train_texts: list,
        train_labels: list,
        val_texts: list,
        val_labels: list,
    ) -> Tuple[float, np.ndarray]:
        """
        Trains model on a single fold, checkpoints best weights, and returns OOF probabilities.
        """
        print(f"\n{'='*20} Starting Fold {fold + 1}/{self.train_cfg.n_splits} {'='*20}")

        # Initialize fresh model & tokenizer for this fold
        model, tokenizer = get_model_and_tokenizer(
            self.model_cfg,
            offline_mode=self.train_cfg.offline_mode,
            local_dir=self.train_cfg.local_weights_dir,
        )
        model.to(self.device)

        # Build Datasets and DataLoaders
        train_dataset = TextDataset(train_texts, train_labels, tokenizer, self.model_cfg.max_length)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer, self.model_cfg.max_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_steps = (len(train_loader) // self.train_cfg.gradient_accumulation_steps) * self.train_cfg.epochs
        optimizer, scheduler = self._get_optimizer_and_scheduler(model, total_steps)

        # Mixed Precision Scaler
        use_amp = self.train_cfg.fp16 and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_auc = 0.0
        best_oof_probs = None
        patience_counter = 0
        best_model_path = os.path.join(self.train_cfg.output_dir, f"model_fold_{fold}.pt")

        for epoch in range(self.train_cfg.epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )
                    loss = outputs["loss"]
                    if self.train_cfg.gradient_accumulation_steps > 1:
                        loss = loss / self.train_cfg.gradient_accumulation_steps

                scaler.scale(loss).backward()
                running_loss += loss.item() * self.train_cfg.gradient_accumulation_steps

                if (step + 1) % self.train_cfg.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            # End of epoch evaluation
            val_metrics, val_probs = self.evaluate(model, val_loader)
            val_auc = val_metrics.get("roc_auc", 0.0)
            avg_train_loss = running_loss / len(train_loader)

            print(
                f"Epoch {epoch + 1}/{self.train_cfg.epochs} - "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val ROC-AUC: {val_auc:.5f} | "
                f"Val F1: {val_metrics.get('f1_macro', 0.0):.4f}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_oof_probs = val_probs
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> [Saved best model checkpoint to {best_model_path}]")
            else:
                patience_counter += 1
                if patience_counter >= self.train_cfg.early_stopping_patience:
                    print(f"Early stopping triggered for Fold {fold + 1}.")
                    break

        # Save tokenizer once
        tokenizer.save_pretrained(self.train_cfg.output_dir)

        # Automatic Red-Teaming stress test on the best fold checkpoint
        if self.red_team_callback is not None:
            try:
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path, map_location=self.device))

                def fold_predict_fn(texts: list) -> np.ndarray:
                    model.eval()
                    ds = TextDataset(texts, tokenizer=tokenizer, max_length=self.model_cfg.max_length)
                    dl = DataLoader(ds, batch_size=self.train_cfg.eval_batch_size, shuffle=False, num_workers=0)
                    _, probs = self.evaluate(model, dl)
                    return probs

                self.red_team_callback.on_fold_end(
                    fold=fold,
                    predict_fn=fold_predict_fn,
                    val_texts=val_texts,
                    val_labels=val_labels,
                )
            except Exception as e:
                print(f"  ⚠️  [Red-Teaming Callback Warning]: {e}")

        # Cleanup memory
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val_auc, best_oof_probs

    def fit_cv(
        self,
        df: Any,
        text_col: str,
        target_col: str,
    ) -> Dict[str, Any]:
        """
        Executes Stratified 5-Fold Cross Validation on the dataset.
        Computes Out-of-Fold (OOF) predictions and overall CV ROC-AUC.
        """
        if "fold" not in df.columns:
            df = create_stratified_folds(
                df,
                target_col=target_col,
                n_splits=self.train_cfg.n_splits,
                seed=self.train_cfg.seed,
            )

        oof_predictions = np.zeros(len(df))
        fold_scores = []

        for fold in range(self.train_cfg.n_splits):
            train_mask = (df["fold"] != fold)
            val_mask = (df["fold"] == fold)

            train_texts = df.loc[train_mask, text_col].tolist()
            train_labels = df.loc[train_mask, target_col].tolist()
            val_texts = df.loc[val_mask, text_col].tolist()
            val_labels = df.loc[val_mask, target_col].tolist()

            best_auc, val_probs = self.train_fold(
                fold=fold,
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
            )
            fold_scores.append(best_auc)
            oof_predictions[val_mask] = val_probs

        overall_auc = compute_roc_auc(df[target_col].to_numpy(), oof_predictions)
        print(f"\n{'*'*50}")
        print(f"5-Fold CV Completed!")
        print(f"Fold AUCs: {[round(s, 5) for s in fold_scores]}")
        print(f"Overall OOF ROC-AUC: {overall_auc:.5f}")
        print(f"{'*'*50}\n")

        # Save config and OOF predictions
        self.train_cfg.save_json(os.path.join(self.train_cfg.output_dir, "train_config.json"))

        return {
            "fold_scores": fold_scores,
            "overall_oof_auc": overall_auc,
            "oof_predictions": oof_predictions,
        }
