"""
Metrics module for evaluation analytics.

Computes:
1. Win rates & counts (Groq, Gemini, Tie)
2. Per-criterion mean scores across 5 criteria + overall
3. Score Clustering analysis (mean, standard deviation, score spread, identical score count)
4. Position Bias analysis (flip count, flip rate)
5. Verbosity / Length analysis (mean lengths, longer-answer preference rate)
"""
import math
from typing import List, Dict, Any


def compute_suite_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes comprehensive aggregate metrics from per-question evaluation results.
    """
    total = len(results)
    if total == 0:
        return _empty_metrics()

    # Winner counts
    groq_wins = sum(1 for r in results if r["winner"] == "groq")
    gemini_wins = sum(1 for r in results if r["winner"] == "gemini")
    ties = sum(1 for r in results if r["winner"] == "tie")
    flips = sum(1 for r in results if r.get("position_flipped", False))

    # Per-criterion means
    criteria = ["correctness", "faithfulness", "completeness", "instruction_following", "tone_safety", "overall"]
    groq_criterion_means = {}
    gemini_criterion_means = {}

    for c in criteria:
        groq_vals = [r["groq_scores"][c] for r in results if "groq_scores" in r]
        gemini_vals = [r["gemini_scores"][c] for r in results if "gemini_scores" in r]
        groq_criterion_means[c] = round(sum(groq_vals) / len(groq_vals), 4) if groq_vals else 0.0
        gemini_criterion_means[c] = round(sum(gemini_vals) / len(gemini_vals), 4) if gemini_vals else 0.0

    # Score Clustering Analysis across all overall scores
    all_overalls = []
    for r in results:
        if "groq_scores" in r:
            all_overalls.append(r["groq_scores"]["overall"])
        if "gemini_scores" in r:
            all_overalls.append(r["gemini_scores"]["overall"])

    if all_overalls:
        mean_all = sum(all_overalls) / len(all_overalls)
        variance = sum((x - mean_all) ** 2 for x in all_overalls) / len(all_overalls)
        std_dev = math.sqrt(variance)
        min_score = min(all_overalls)
        max_score = max(all_overalls)
        score_counts = {}
        for s in all_overalls:
            score_counts[s] = score_counts.get(s, 0) + 1
        max_identical_count = max(score_counts.values()) if score_counts else 0
    else:
        mean_all = 0.0
        std_dev = 0.0
        min_score = 0.0
        max_score = 0.0
        max_identical_count = 0

    # Verbosity / Length Analysis
    groq_lens = [r.get("groq_char_length", 0) for r in results]
    gemini_lens = [r.get("gemini_char_length", 0) for r in results]
    avg_groq_len = round(sum(groq_lens) / total) if groq_lens else 0
    avg_gemini_len = round(sum(gemini_lens) / total) if gemini_lens else 0

    longer_wins = 0
    decisive_cases = 0
    for r in results:
        g_len = r.get("groq_char_length", 0)
        m_len = r.get("gemini_char_length", 0)
        winner = r.get("winner", "")
        if winner == "groq":
            decisive_cases += 1
            if g_len > m_len:
                longer_wins += 1
        elif winner == "gemini":
            decisive_cases += 1
            if m_len > g_len:
                longer_wins += 1

    longer_answer_win_rate = round(longer_wins / decisive_cases, 4) if decisive_cases > 0 else 0.0

    # Latency & Token Usage
    latencies = [r.get("latency_ms", 0) for r in results]
    avg_latency = round(sum(latencies) / total) if latencies else 0
    total_tokens = sum(r.get("total_tokens", 0) for r in results)

    return {
        "total_questions": total,
        "winner_counts": {
            "groq": groq_wins,
            "gemini": gemini_wins,
            "tie": ties,
        },
        "win_rates": {
            "groq": round(groq_wins / total, 4),
            "gemini": round(gemini_wins / total, 4),
            "tie": round(ties / total, 4),
        },
        "mean_scores": {
            "groq": groq_criterion_means,
            "gemini": gemini_criterion_means,
        },
        "score_clustering": {
            "mean_overall": round(mean_all, 4),
            "std_dev": round(std_dev, 4),
            "score_min": min_score,
            "score_max": max_score,
            "score_range": round(max_score - min_score, 4),
            "max_identical_score_count": max_identical_count,
        },
        "position_bias": {
            "flip_count": flips,
            "flip_rate": round(flips / total, 4),
        },
        "verbosity_length": {
            "avg_groq_char_length": avg_groq_len,
            "avg_gemini_char_length": avg_gemini_len,
            "longer_answer_win_rate": longer_answer_win_rate,
        },
        "latency": {
            "average_ms": avg_latency,
        },
        "total_tokens": total_tokens,
    }


def _empty_metrics() -> Dict[str, Any]:
    criteria = {c: 0.0 for c in ["correctness", "faithfulness", "completeness", "instruction_following", "tone_safety", "overall"]}
    return {
        "total_questions": 0,
        "winner_counts": {"groq": 0, "gemini": 0, "tie": 0},
        "win_rates": {"groq": 0.0, "gemini": 0.0, "tie": 0.0},
        "mean_scores": {"groq": criteria.copy(), "gemini": criteria.copy()},
        "score_clustering": {"mean_overall": 0.0, "std_dev": 0.0, "score_min": 0.0, "score_max": 0.0, "score_range": 0.0, "max_identical_score_count": 0},
        "position_bias": {"flip_count": 0, "flip_rate": 0.0},
        "verbosity_length": {"avg_groq_char_length": 0, "avg_gemini_char_length": 0, "longer_answer_win_rate": 0.0},
        "latency": {"average_ms": 0},
        "total_tokens": 0,
    }
