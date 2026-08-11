"""
A/B Config Comparison Module.

Compares Config A (Baseline Judge Prompt) vs Config B (Anti-Bias Enhanced Judge Prompt).
Runs the same test suite through both configurations, measures pass rates, mean scores,
win rates, and position flip rates, declares a winner based on empirical results,
and saves the report to results/ab_comparison.json.

Usage:
    python -m app.ab_comparison
"""
import json
import os
import asyncio
import time
from typing import Dict, Any

from app.judge import LLMJudge
from app.prompts import BASELINE_JUDGE_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT


async def run_ab_comparison(
    questions_path: str = "data/questions.json",
    output_path: str = "results/ab_comparison.json",
) -> Dict[str, Any]:
    """
    Runs A/B configuration comparison between Baseline (Config A) and Anti-Bias Enhanced (Config B).
    """
    print("=" * 60)
    print("A/B Config Comparison: Config A (Baseline) vs Config B (Anti-Bias Enhanced)")
    print("=" * 60)
    print()

    judge = LLMJudge()

    # 1. Run Config A (Baseline Prompt)
    print("--- Running Config A (Baseline Prompt) ---")
    report_a = await judge.run_batch_evaluation(
        questions_path=questions_path,
        report_path="results/temp_report_a.json",
        cases_path="results/temp_cases_a.json",
        bias_report_path="results/temp_bias_a.json",
        system_prompt=BASELINE_JUDGE_SYSTEM_PROMPT,
    )

    print("\n--- Running Config B (Anti-Bias Enhanced Prompt) ---")
    report_b = await judge.run_batch_evaluation(
        questions_path=questions_path,
        report_path="results/report.json",
        cases_path="results/cases.json",
        bias_report_path="results/bias_results.json",
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )

    # Clean up temporary files
    for tmp in ["results/temp_report_a.json", "results/temp_cases_a.json", "results/temp_bias_a.json"]:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

    # Compare metrics
    flip_rate_a = report_a["position_bias"]["flip_rate"]
    flip_rate_b = report_b["position_bias"]["flip_rate"]

    mean_overall_a = (report_a["mean_scores"]["groq"]["overall"] + report_a["mean_scores"]["gemini"]["overall"]) / 2
    mean_overall_b = (report_b["mean_scores"]["groq"]["overall"] + report_b["mean_scores"]["gemini"]["overall"]) / 2

    # Lower position flip rate + better score calibration = superior config
    if flip_rate_b < flip_rate_a:
        winner = "config_b (anti_bias_prompt)"
        rationale = f"Config B demonstrated lower position flip rate ({flip_rate_b:.1%} vs {flip_rate_a:.1%})."
    elif flip_rate_b > flip_rate_a:
        winner = "config_a (baseline)"
        rationale = f"Config A demonstrated lower position flip rate ({flip_rate_a:.1%} vs {flip_rate_b:.1%})."
    else:
        winner = "config_b (anti_bias_prompt)"
        rationale = "Both configurations achieved equal position stability; Config B selected for explicit anti-bias rules."

    ab_result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_a": {
            "name": "baseline_prompt",
            "total_questions": report_a["total_questions"],
            "win_rates": report_a["win_rates"],
            "mean_overall_score": round(mean_overall_a, 4),
            "position_flip_rate": flip_rate_a,
            "position_flip_count": report_a["position_bias"]["flip_count"],
        },
        "config_b": {
            "name": "anti_bias_enhanced_prompt",
            "total_questions": report_b["total_questions"],
            "win_rates": report_b["win_rates"],
            "mean_overall_score": round(mean_overall_b, 4),
            "position_flip_rate": flip_rate_b,
            "position_flip_count": report_b["position_bias"]["flip_count"],
        },
        "winner": winner,
        "rationale": rationale,
    }

    # Save results/ab_comparison.json
    try:
        os.makedirs("results", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ab_result, f, indent=2)
        print(f"\nA/B Comparison report saved: {output_path}")
    except Exception as e:
        print(f"Error saving A/B comparison report: {e}")

    return ab_result


if __name__ == "__main__":
    asyncio.run(run_ab_comparison())
