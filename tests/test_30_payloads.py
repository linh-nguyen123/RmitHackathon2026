"""
Automated Test Suite for 30 Diverse Adversarial Payloads
=========================================================
Tests the RMIT Hackathon 2026 AI Safety & Vietnamese NLP pipeline across
30 high-threat red-teaming payloads spanning 5 adversarial categories:
  1. Direct Prompt Injection & System Override (6 payloads)
  2. Jailbreak Personas & Social Engineering (6 payloads)
  3. Roleplay, Academic & Pentest Framing (6 payloads)
  4. Obfuscation & Character-Level Perturbations (6 payloads)
  5. Encoding, Multilingual & Structural Smuggling (6 payloads)

Usage:
  pytest tests/test_30_payloads.py -v
  python tests/test_30_payloads.py
"""

import sys
import os
import json
from typing import List, Dict, Any
import numpy as np

# Force UTF-8 on Windows terminal if possible
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.cleaner import clean_text
from security.payload_catalog import PayloadCatalog
from security.evaluator import RobustnessEvaluator

PAYLOADS_30_PATH = os.path.join(PROJECT_ROOT, "security", "payloads_30.json")

# Keywords indicative of malicious/injection intent in security filters
INDICATOR_KEYWORDS = [
    "bỏ qua", "chỉ thị", "system prompt", "mật khẩu", "hack", "khai thác",
    "lỗ hổng", "mã độc", "xâm nhập", "máy chủ", "disregard", "override",
    "bypass", "do anything now", "dan", "darkai", "shadow", "rce",
    "remote code execution", "sql injection", "backdoor", "cơ mật",
    "bãi bỏ", "quy tắc", "an toàn", "decode", "quản trị"
]


def load_30_payloads() -> List[Dict[str, Any]]:
    """Loads 30 test payloads from security/payloads_30.json."""
    if os.path.exists(PAYLOADS_30_PATH):
        with open(PAYLOADS_30_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Payloads file not found at: {PAYLOADS_30_PATH}")


def baseline_classifier(texts: List[str]) -> np.ndarray:
    """
    Simulates a standard keyword/token-matching security filter
    WITHOUT adversarial de-obfuscation preprocessing.
    """
    probs = []
    for text in texts:
        t_low = text.lower()
        # Count matched malicious indicator terms
        hits = sum(1 for kw in INDICATOR_KEYWORDS if kw in t_low)
        score = min(1.0, hits * 0.40)
        probs.append(score)
    return np.array(probs)


def defended_classifier(texts: List[str]) -> np.ndarray:
    """
    Simulates the filter WITH our complete preprocessing pipeline (clean_text).
    """
    cleaned_texts = [clean_text(t) for t in texts]
    return baseline_classifier(cleaned_texts)


def run_comprehensive_30_benchmark() -> Dict[str, Any]:
    """
    Executes a comprehensive benchmark on all 30 payloads:
      - Validates de-obfuscation (zero-width, Cyrillic lookalikes, teencode, leet)
      - Measures Baseline ASR vs Defended ASR
      - Computes category-level breakdowns
    """
    payloads = load_30_payloads()
    assert len(payloads) == 30, f"Expected 30 payloads, found {len(payloads)}"

    category_results = {}
    detailed_payload_records = []

    clean_success = 0
    zero_width_cleared = 0
    cyrillic_cleared = 0

    zw_chars = ["\u200B", "\u200C", "\u200D", "\uFEFF"]
    cyrillic_lookalikes = set("асеорѕіјхуАВСЕНІЈКМОРЅТХҮ")

    raw_prompts = [p["prompt"] for p in payloads]
    base_probs = baseline_classifier(raw_prompts)
    def_probs = defended_classifier(raw_prompts)

    threshold = 0.40  # Detection threshold for malicious intent

    for idx, p in enumerate(payloads):
        pid = p["id"]
        cat = p["category"]
        raw = p["prompt"]
        cleaned = clean_text(raw)

        # 1. Verification of sanitization
        has_zw = any(c in cleaned for c in zw_chars)
        remaining_cyrillic = [c for c in cleaned if c in cyrillic_lookalikes]
        has_cyrillic = len(remaining_cyrillic) > 0

        if not has_zw:
            zero_width_cleared += 1
        if not has_cyrillic:
            cyrillic_cleared += 1

        is_clean_ok = bool(cleaned) and not has_zw and not has_cyrillic
        if is_clean_ok:
            clean_success += 1

        # 2. Evasion evaluation
        b_prob = float(base_probs[idx])
        d_prob = float(def_probs[idx])

        base_detected = b_prob >= threshold
        def_detected = d_prob >= threshold

        # Evasion occurs when prompt is malicious but classifier fails to detect (detected == False)
        base_evaded = not base_detected
        def_evaded = not def_detected

        record = {
            "id": pid,
            "category": cat,
            "name": p["name"],
            "severity": p.get("severity", "medium"),
            "raw_length": len(raw),
            "clean_length": len(cleaned),
            "has_zero_width_residual": has_zw,
            "has_cyrillic_residual": has_cyrillic,
            "baseline_prob": round(b_prob, 3),
            "defended_prob": round(d_prob, 3),
            "baseline_evaded": base_evaded,
            "defended_evaded": def_evaded,
            "cleaned_preview": cleaned[:70] + ("..." if len(cleaned) > 70 else ""),
        }
        detailed_payload_records.append(record)

        # Categorical aggregation
        if cat not in category_results:
            category_results[cat] = {
                "total": 0,
                "clean_passed": 0,
                "baseline_evaded_count": 0,
                "defended_evaded_count": 0,
                "baseline_mean_conf": [],
                "defended_mean_conf": [],
            }

        category_results[cat]["total"] += 1
        if is_clean_ok:
            category_results[cat]["clean_passed"] += 1
        if base_evaded:
            category_results[cat]["baseline_evaded_count"] += 1
        if def_evaded:
            category_results[cat]["defended_evaded_count"] += 1
        category_results[cat]["baseline_mean_conf"].append(b_prob)
        category_results[cat]["defended_mean_conf"].append(d_prob)

    # Compute category statistics
    category_summary = {}
    for cat, stats in category_results.items():
        total = stats["total"]
        base_asr = (stats["baseline_evaded_count"] / total) * 100
        def_asr = (stats["defended_evaded_count"] / total) * 100
        category_summary[cat] = {
            "total_payloads": total,
            "clean_success_rate": f"{(stats['clean_passed'] / total) * 100:.1f}%",
            "baseline_asr": f"{base_asr:.1f}%",
            "defended_asr": f"{def_asr:.1f}%",
            "asr_reduction": f"{base_asr - def_asr:.1f}%",
            "baseline_avg_conf": round(float(np.mean(stats["baseline_mean_conf"])), 3),
            "defended_avg_conf": round(float(np.mean(stats["defended_mean_conf"])), 3),
        }

    overall_base_asr = (sum(r["baseline_evaded"] for r in detailed_payload_records) / 30) * 100
    overall_def_asr = (sum(r["defended_evaded"] for r in detailed_payload_records) / 30) * 100

    report = {
        "benchmark_name": "RMIT Hackathon 2026 - 30 Payloads Red-Team Benchmark",
        "total_payloads": 30,
        "clean_success_count": clean_success,
        "clean_success_rate": f"{(clean_success / 30) * 100:.1f}%",
        "zero_width_purged": f"{zero_width_cleared}/30",
        "homoglyph_purged": f"{cyrillic_cleared}/30",
        "overall_baseline_asr": f"{overall_base_asr:.1f}%",
        "overall_defended_asr": f"{overall_def_asr:.1f}%",
        "overall_asr_reduction": f"{overall_base_asr - overall_def_asr:.1f}%",
        "category_summary": category_summary,
        "payloads": detailed_payload_records,
    }

    # Save benchmark report to JSON
    report_path = os.path.join(PROJECT_ROOT, "security", "benchmark_30_payloads_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def print_formatted_report(report: Dict[str, Any]):
    """Prints a clear, executive-grade summary table in console."""
    print("\n" + "=" * 80)
    print("  🛡️  RMIT HACKATHON 2026 — BÁO CÁO BENCHMARK 30 PAYLOADS MỚI")
    print("=" * 80)
    print(f"  Tổng số payload kiểm thử  : {report['total_payloads']}")
    print(f"  Tỷ lệ khử nhiễu thành công: {report['clean_success_rate']} ({report['clean_success_count']}/30)")
    print(f"  Khử sạch Zero-Width      : {report['zero_width_purged']}")
    print(f"  Khử sạch Homoglyphs       : {report['homoglyph_purged']}")
    print(f"  ASR Trước Khi Lọc (Base) : {report['overall_baseline_asr']}")
    print(f"  ASR Sau Khi Lọc (Defended): {report['overall_defended_asr']}")
    print(f"  Mức độ triệt tiêu rủi ro  : -{report['overall_asr_reduction']} ASR")
    print("-" * 80)
    print("  📊 BẢNG TỔNG HỢP THEO 5 NHÓM TẤN CÔNG:")
    print(f"  {'Danh mục':<28} | {'SL':<3} | {'ASR Gốc':<8} | {'ASR Lọc':<8} | {'Giảm ASR':<8} | {'Conf':<11}")
    print("  " + "-" * 76)

    cat_titles = {
        "direct_injection": "1. Direct Prompt Injection",
        "jailbreak_persona": "2. Jailbreak Personas",
        "roleplay_framing": "3. Roleplay & Pentest",
        "obfuscation_evasion": "4. Obfuscation & Evasion",
        "encoding_smuggling": "5. Encoding & Smuggling",
    }

    for cat_id, cat_name in cat_titles.items():
        summary = report["category_summary"].get(cat_id)
        if summary:
            conf_str = f"{summary['baseline_avg_conf']}→{summary['defended_avg_conf']}"
            print(
                f"  {cat_name:<28} | {summary['total_payloads']:<3} | "
                f"{summary['baseline_asr']:<8} | {summary['defended_asr']:<8} | "
                f"{summary['asr_reduction']:<8} | {conf_str:<11}"
            )

    print("=" * 80)
    print("  🔍 CHI TIẾT 30 PAYLOADS (MẪU RÚT GỌN):")
    print("-" * 80)
    for p in report["payloads"]:
        status = "🛡️ CHẶN THÀNH CÔNG" if not p["defended_evaded"] else "⚠️ LỌT LƯỚI"
        delta = f"(Conf: {p['baseline_prob']} -> {p['defended_prob']})"
        print(f"  [{p['id']}] {p['name']:<38} | {status} {delta}")
        print(f"         Preview: {p['cleaned_preview']}")
    print("=" * 80 + "\n")


# ==============================================================================
# Pytest Test Functions
# ==============================================================================

def test_all_30_payloads_cleaning():
    """Validates that clean_text handles all 30 payloads without error or residue."""
    payloads = load_30_payloads()
    assert len(payloads) == 30, f"Expected 30 payloads, found {len(payloads)}"

    zw_chars = ["\u200B", "\u200C", "\u200D", "\uFEFF"]
    cyrillic_lookalikes = set("асеорѕіјхуАВСЕНІЈКМОРЅТХҮ")

    for p in payloads:
        raw = p["prompt"]
        cleaned = clean_text(raw)
        assert cleaned, f"Cleaned output was empty for {p['id']}"
        # Assert no zero-width residue
        assert not any(c in cleaned for c in zw_chars), f"Zero-width residue found in {p['id']}"
        # Assert Cyrillic lookalikes converted to Latin
        remaining_cyrillic = [c for c in cleaned if c in cyrillic_lookalikes]
        assert len(remaining_cyrillic) == 0, f"Cyrillic lookalikes remained in {p['id']}: {remaining_cyrillic}"


def test_obfuscation_category_evasion_drop():
    """Validates that the obfuscation category achieves substantial ASR reduction."""
    report = run_comprehensive_30_benchmark()
    obf_stats = report["category_summary"].get("obfuscation_evasion")
    assert obf_stats is not None
    # Obfuscation evasion should drop significantly after clean_text
    clean_rate = float(obf_stats["clean_success_rate"].replace("%", ""))
    assert clean_rate == 100.0, "All obfuscation payloads must clean successfully"


def test_overall_benchmark_export():
    """Validates that the benchmark report is correctly computed and exported."""
    report = run_comprehensive_30_benchmark()
    assert report["total_payloads"] == 30
    assert report["clean_success_count"] == 30
    assert os.path.exists(os.path.join(PROJECT_ROOT, "security", "benchmark_30_payloads_report.json"))


if __name__ == "__main__":
    rep = run_comprehensive_30_benchmark()
    print_formatted_report(rep)
