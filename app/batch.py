"""
CLI entry point for batch evaluation.

Usage:
    python -m app.batch

Runs the full evaluation suite against data/questions.json,
performs A/B + B/A bias testing per question, computes aggregate metrics,
and saves results to results/report.json + results/cases.json.
"""
import asyncio
from app.judge import LLMJudge


async def main():
    print("=" * 60)
    print("LLM-as-Judge — Batch Evaluation")
    print("=" * 60)
    print()

    judge = LLMJudge()
    report = await judge.run_batch_evaluation()

    print()
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Total questions: {report['total_questions']}")
    print(f"Groq wins:  {report['winner_counts']['groq']}  ({report['win_rates']['groq']:.0%})")
    print(f"Gemini wins: {report['winner_counts']['gemini']}  ({report['win_rates']['gemini']:.0%})")
    print(f"Ties:       {report['winner_counts']['tie']}  ({report['win_rates']['tie']:.0%})")
    print()
    print(f"Mean Groq overall:   {report['mean_scores']['groq']['overall']}")
    print(f"Mean Gemini overall: {report['mean_scores']['gemini']['overall']}")
    print()
    print(f"Position flips: {report['position_bias']['flip_count']} ({report['position_bias']['flip_rate']:.0%})")
    print(f"Average latency: {report['latency']['average_ms']}ms")
    print(f"Total tokens: {report['total_tokens']}")
    print(f"Total time: {report['total_latency_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
