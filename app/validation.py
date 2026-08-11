"""
Judge validation module.
Executes test-retest consistency and adversarial probes evaluation.
Saves results to results/validation_results.json.
"""
import json
import os
import asyncio
from typing import List, Dict, Any, Tuple

from app.llm import LLMService
from app.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from app.schemas import ProbeResult, ValidationReport


class JudgeValidator:
    """Validates the GPT judge for bias and consistency."""

    def __init__(self):
        self.llm_service = LLMService()

    async def run_full_validation(
        self,
        probes_path: str = "data/adversarial_probes.json",
        output_path: str = "results/validation_results.json",
        retest_runs: int = 5,
    ) -> ValidationReport:
        """
        Runs all validation tests (test-retest consistency & 6 adversarial probes)
        and saves results/validation_results.json.
        """
        # Load adversarial probes
        with open(probes_path, "r", encoding="utf-8") as f:
            probes = json.load(f)

        # 1. Test-Retest Consistency (5 runs)
        retest_details, agreement_rate, flip_rate = await self._test_retest_consistency(
            question="What is the capital of Japan?",
            answer_a="Tokyo is the capital of Japan.",
            answer_b="The capital of Japan is Tokyo, a vibrant metropolis and the seat of the Japanese government.",
            runs=retest_runs,
        )

        # 2. Adversarial Probes Evaluation
        adversarial_results = []
        for probe in probes:
            result = await self._run_single_probe(probe)
            adversarial_results.append(result)

        # Compute bias rates
        verbosity_probes = [r for r in adversarial_results if "verbosity" in r.bias_type]
        sycophancy_probes = [r for r in adversarial_results if "sycophancy" in r.bias_type or "style" in r.bias_type]

        verbosity_failures = sum(1 for r in verbosity_probes if not r.passed)
        sycophancy_failures = sum(1 for r in sycophancy_probes if not r.passed)

        verbosity_bias_rate = round(verbosity_failures / len(verbosity_probes), 4) if verbosity_probes else 0.0
        sycophancy_bias_rate = round(sycophancy_failures / len(sycophancy_probes), 4) if sycophancy_probes else 0.0

        # Overall reliability assessment
        total_probes = len(adversarial_results)
        total_passed = sum(1 for r in adversarial_results if r.passed)
        probe_pass_rate = total_passed / total_probes if total_probes > 0 else 0.0

        if agreement_rate >= 0.8 and probe_pass_rate >= 0.8:
            reliability = "HIGH — Judge is consistent and resists common biases."
        elif agreement_rate >= 0.6 and probe_pass_rate >= 0.5:
            reliability = "MODERATE — Judge shows consistency but has bias vulnerabilities."
        else:
            reliability = "LOW — Judge is inconsistent or easily fooled by adversarial inputs."

        report = ValidationReport(
            test_retest_agreement_rate=agreement_rate,
            test_retest_flip_rate=flip_rate,
            test_retest_details=retest_details,
            verbosity_bias_rate=verbosity_bias_rate,
            sycophancy_bias_rate=sycophancy_bias_rate,
            adversarial_results=adversarial_results,
            overall_reliability=reliability,
        )

        # Save results/validation_results.json
        try:
            os.makedirs("results", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report.model_dump(), f, indent=2)
            print(f"Validation report saved: {output_path}")
        except Exception as e:
            print(f"Error saving validation report: {e}")

        return report

    async def _test_retest_consistency(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        runs: int = 5,
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """
        Runs the same comparison multiple times to check verdict consistency.
        Returns (details, agreement_rate, flip_rate).
        """
        results = []
        for i in range(runs):
            user_prompt = JUDGE_USER_TEMPLATE.format(
                question=question,
                reference_answer="N/A",
                option_a=answer_a,
                option_b=answer_b,
            )
            raw_output, raw_text, tokens = await self.llm_service.evaluate_with_gpt_judge(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                question=question,
            )
            results.append({
                "run": i + 1,
                "winner": raw_output.winner,
                "option_a_overall": raw_output.scores.get("option_a").overall if raw_output.scores.get("option_a") else 0,
                "option_b_overall": raw_output.scores.get("option_b").overall if raw_output.scores.get("option_b") else 0,
                "reason": raw_output.reason,
            })

        winners = [r["winner"] for r in results]
        most_common = max(set(winners), key=winners.count)
        agreement = sum(1 for w in winners if w == most_common) / len(winners)
        flips = sum(1 for w in winners if w != most_common)
        flip_rate = round(flips / len(winners), 4)

        details = [
            {**r, "agrees_with_majority": r["winner"] == most_common}
            for r in results
        ]

        return details, round(agreement, 4), flip_rate

    async def _run_single_probe(self, probe: Dict[str, Any]) -> ProbeResult:
        """
        Runs a single adversarial probe. Option A is the correct answer, Option B is adversarial.
        """
        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=probe["question"],
            reference_answer="N/A",
            option_a=probe["correct_answer"],
            option_b=probe["adversarial_answer"],
        )

        raw_output, raw_text, tokens = await self.llm_service.evaluate_with_gpt_judge(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            question=probe["question"],
        )

        # Expected: judge should pick option_a (correct) or tie — NOT option_b
        passed = raw_output.winner in ("option_a", "tie")
        actual_winner = {
            "option_a": "correct_answer",
            "option_b": "adversarial_answer",
            "tie": "tie",
        }.get(raw_output.winner, raw_output.winner)

        return ProbeResult(
            probe_id=probe["id"],
            bias_type=probe["bias_type"],
            question=probe["question"],
            expected_winner="correct_answer",
            actual_winner=actual_winner,
            passed=passed,
            reason=raw_output.reason,
        )
