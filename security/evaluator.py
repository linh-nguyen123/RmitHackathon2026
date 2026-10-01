"""
Robustness Evaluator & Attack Success Rate (ASR) Benchmark.
Measures classifier resistance against adversarial mutations and prompt injections.
"""

import json
from typing import Callable, List, Dict, Any, Optional
import numpy as np
try:
    from modeling.metrics import compute_roc_auc
except (ImportError, ValueError):
    try:
        from ..modeling.metrics import compute_roc_auc
    except (ImportError, ValueError):
        from sklearn.metrics import roc_auc_score
        def compute_roc_auc(y_true, y_pred, **kwargs):
            try:
                return float(roc_auc_score(y_true, y_pred))
            except Exception:
                return 0.5
from .perturbations import AdversarialPerturber


class RobustnessEvaluator:
    """
    Evaluates classifier vulnerability to adversarial evasion attacks.
    Calculates Attack Success Rate (ASR) and ROC-AUC degradation.
    """

    def __init__(self, predict_fn: Callable[[List[str]], np.ndarray]):
        """
        Args:
            predict_fn: A function that takes a list of strings and returns predicted
                        probabilities (1D array of positive class probabilities).
        """
        self.predict_fn = predict_fn

    def evaluate_attack_success_rate(
        self,
        clean_texts: List[str],
        perturbed_texts: List[str],
        ground_truth: Optional[List[int]] = None,
        threshold: float = 0.5,
        target_class: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculates Attack Success Rate (ASR).
        For security filters, ASR represents the percentage of malicious prompts that
        successfully evade detection (prediction flips from target_class to benign).
        """
        clean_probs = np.asarray(self.predict_fn(clean_texts))
        perturbed_probs = np.asarray(self.predict_fn(perturbed_texts))

        # Flatten if 2D
        if clean_probs.ndim == 2 and clean_probs.shape[1] == 2:
            clean_probs = clean_probs[:, 1]
        if perturbed_probs.ndim == 2 and perturbed_probs.shape[1] == 2:
            perturbed_probs = perturbed_probs[:, 1]

        clean_preds = (clean_probs >= threshold).astype(int)
        perturbed_preds = (perturbed_probs >= threshold).astype(int)

        # Attacks that successfully flipped prediction from toxic/malicious (1) to safe (0)
        # Evasion success: clean was detected (1), perturbed bypassed detection (0)
        detected_indices = np.where(clean_preds == target_class)[0]
        if len(detected_indices) > 0:
            bypassed = np.sum(perturbed_preds[detected_indices] != target_class)
            asr = float(bypassed / len(detected_indices))
        else:
            asr = 0.0

        avg_confidence_drop = float(np.mean(clean_probs - perturbed_probs))

        report = {
            "total_samples": len(clean_texts),
            "clean_detected_count": int(len(detected_indices)),
            "attack_success_rate": round(asr, 4),
            "evasion_percentage": f"{asr * 100:.2f}%",
            "avg_confidence_drop": round(avg_confidence_drop, 4),
            "mean_clean_prob": round(float(np.mean(clean_probs)), 4),
            "mean_perturbed_prob": round(float(np.mean(perturbed_probs)), 4),
        }

        # If ground truth is provided, compute ROC-AUC drop
        if ground_truth is not None:
            clean_auc = compute_roc_auc(ground_truth, clean_probs)
            perturbed_auc = compute_roc_auc(ground_truth, perturbed_probs)
            report["clean_roc_auc"] = round(clean_auc, 5)
            report["perturbed_roc_auc"] = round(perturbed_auc, 5)
            report["roc_auc_drop"] = round(clean_auc - perturbed_auc, 5)

        return report

    def benchmark_adversarial_suite(
        self,
        samples: List[str],
        ground_truth: Optional[List[int]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Runs a comprehensive benchmark across all mutation types:
        zero_width, homoglyph, teencode, leetspeak, dan, compound_attack.
        """
        mutation_types = [
            "zero_width", "homoglyph", "teencode", "leetspeak",
            "multilingual_dan", "compound_attack"
        ]

        suite_results = {}
        for m_type in mutation_types:
            perturbed_samples = []
            for text in samples:
                variants = AdversarialPerturber.generate_adversarial_suite(text)
                perturbed_samples.append(variants.get(m_type, text))

            eval_res = self.evaluate_attack_success_rate(
                clean_texts=samples,
                perturbed_texts=perturbed_samples,
                ground_truth=ground_truth,
                threshold=threshold,
            )
            suite_results[m_type] = eval_res

        return suite_results

    @staticmethod
    def export_report(report_data: Dict[str, Any], filepath: str = "security_report.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
