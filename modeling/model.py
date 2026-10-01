"""
Transformer Model Architectures & Classification Heads.
Supports Mean Pooling, Multi-Sample Dropout, and Offline Loading for Kaggle.
"""

from typing import Optional, Tuple, Any, Dict
import os

try:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:  # type: ignore
        Module = object

from .config import ModelConfig


class MeanPooling(nn.Module if TORCH_AVAILABLE else object):
    """
    Mean Pooling layer that averages token embeddings taking attention_mask into account.
    Significantly more robust than relying solely on the [CLS] representation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled


class MultiSampleDropout(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-Sample Dropout: passes embeddings through multiple dropout masks
    and averages their linear projections. Accelerates convergence and regularizes.
    """
    def __init__(self, in_features: int, out_features: int, dropout_rates=(0.1, 0.2, 0.3, 0.4, 0.5)):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Any) -> Any:
        logits = torch.mean(
            torch.stack([self.linear(drop(x)) for drop in self.dropouts], dim=0),
            dim=0,
        )
        return logits


class TransformerClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    Complete sequence classification model wrapping Hugging Face backbones
    (such as microsoft/mdeberta-v3-base or vinai/phobert-base-v2).
    """

    def __init__(self, config: ModelConfig, pretrained_model_or_path: Optional[str] = None):
        super().__init__()
        self.model_cfg = config
        path_to_load = pretrained_model_or_path or config.model_name_or_path

        # Load Hugging Face backbone
        self.transformer_config = AutoConfig.from_pretrained(path_to_load)
        self.transformer_config.update({
            "output_hidden_states": True,
            "hidden_dropout_prob": config.dropout_rate,
            "attention_probs_dropout_prob": config.dropout_rate,
        })
        self.backbone = AutoModel.from_pretrained(path_to_load, config=self.transformer_config)

        hidden_size = self.transformer_config.hidden_size
        self.use_mean_pooling = config.use_mean_pooling
        if self.use_mean_pooling:
            self.pooler = MeanPooling()

        # Classification Head
        if config.multi_sample_dropout:
            self.classifier = MultiSampleDropout(hidden_size, config.num_labels)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Linear(hidden_size, config.num_labels),
            )

        self.loss_fn = nn.CrossEntropyLoss() if config.num_labels > 1 else nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Optional[Any] = None,
        labels: Optional[Any] = None,
    ) -> Dict[str, Any]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)
        last_hidden_state = outputs.last_hidden_state

        if self.use_mean_pooling:
            feature = self.pooler(last_hidden_state, attention_mask)
        else:
            # Use [CLS] representation (first token)
            feature = last_hidden_state[:, 0, :]

        logits = self.classifier(feature)

        loss = None
        if labels is not None:
            if self.model_cfg.num_labels == 1:
                loss = self.loss_fn(logits.view(-1), labels.view(-1).float())
            else:
                loss = self.loss_fn(logits.view(-1, self.model_cfg.num_labels), labels.view(-1))

        return {"logits": logits, "loss": loss}


def get_model_and_tokenizer(
    model_cfg: ModelConfig,
    offline_mode: bool = False,
    local_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Loads Hugging Face model and tokenizer.
    If offline_mode is True or local_dir exists, loads strictly from local Kaggle directory.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and Transformers must be installed to initialize models.")

    load_path = local_dir if (offline_mode and local_dir and os.path.exists(local_dir)) else model_cfg.model_name_or_path
    local_files_only = offline_mode or (local_dir is not None and os.path.exists(local_dir))

    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        local_files_only=local_files_only,
        use_fast=True if model_cfg.model_type != "phobert" else False,
    )

    model = TransformerClassifier(model_cfg, pretrained_model_or_path=load_path)

    return model, tokenizer
