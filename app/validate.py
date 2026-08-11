"""
CLI entry point for Judge Validation.

Usage:
    python -m app.validate

Runs test-retest consistency checks and 6 adversarial probes,
saves results/validation_results.json, and prints summary to console.
"""
import asyncio
from app.validation import JudgeValidator


async def main():
    print("=" * 60)
    print("LLM-as-Judge — Judge Validation & Adversarial Probes")
    print("=" * 60)
    print()

    validator = JudgeValidator()
    report = await validator.run_full_validation()

    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Test-Retest Agreement Rate: {report.test_retest_agreement_rate:.0%}")
    print(f"Test-Retest Flip Rate:      {report.test_retest_flip_rate:.0%}")
    print(f"Verbosity Bias Failure Rate: {report.verbosity_bias_rate:.0%}")
    print(f"Sycophancy Bias Failure Rate:{report.sycophancy_bias_rate:.0%}")
    print(f"Overall Reliability Rating:  {report.overall_reliability}")
    print()
    print("Adversarial Probes Breakdown:")
    for res in report.adversarial_results:
        status = "PASSED" if res.passed else "FAILED"
        print(f"  [{res.probe_id}] {res.bias_type:<30} -> {status}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
