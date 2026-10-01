"""
Catalog and Registry for Adversarial Payloads & Red-Teaming Benchmarks.
Manages prompt injection, jailbreak prompts, and multilingual evasion attacks.
"""

import os
import json
from typing import List, Dict, Any, Optional


class PayloadCatalog:
    """
    Catalog manager for security benchmarks and red-teaming datasets.
    """

    DEFAULT_JSON_PATH = os.path.join(os.path.dirname(__file__), "payloads.json")

    def __init__(self, json_path: Optional[str] = None):
        self.json_path = json_path or self.DEFAULT_JSON_PATH
        self.payloads: List[Dict[str, Any]] = self.load_payloads(self.json_path)

    @staticmethod
    def load_payloads(filepath: str) -> List[Dict[str, Any]]:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.payloads)

    def filter_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Filter payloads by category: 'prompt_injection', 'jailbreak_dan', 'roleplay_framing', 'multilingual_evasion', 'obfuscation'."""
        cat_lower = category.lower()
        return [p for p in self.payloads if p.get("category", "").lower() == cat_lower]

    def filter_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Filter by language code: 'vi', 'en', 'bilingual'."""
        lang_lower = language.lower()
        return [p for p in self.payloads if p.get("language", "").lower() == lang_lower]

    def get_by_id(self, payload_id: str) -> Optional[Dict[str, Any]]:
        for p in self.payloads:
            if p.get("id") == payload_id:
                return p
        return None

    def add_payload(
        self,
        name: str,
        prompt: str,
        category: str,
        language: str = "vi",
        severity: str = "medium",
        attack_vector: str = "custom",
    ) -> Dict[str, Any]:
        """Adds a new adversarial payload to the catalog."""
        new_id = f"CUSTOM-{len(self.payloads) + 1:03d}"
        item = {
            "id": new_id,
            "name": name,
            "category": category,
            "language": language,
            "prompt": prompt,
            "severity": severity,
            "attack_vector": attack_vector,
        }
        self.payloads.append(item)
        return item

    def save(self, filepath: Optional[str] = None):
        target = filepath or self.json_path
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.payloads, f, indent=2, ensure_ascii=False)
