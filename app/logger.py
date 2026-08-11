"""
Evaluation and Judge Call Logger — JSONL append mode.

Logs:
1. logs/evaluations.jsonl — User /evaluate requests & batch evaluations.
2. logs/judge_calls.jsonl — Complete auditable prompt, raw GPT response, parsed verdict, model name, tokens.
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


LOG_DIR = "logs"
EVALUATIONS_LOG = os.path.join(LOG_DIR, "evaluations.jsonl")
JUDGE_CALLS_LOG = os.path.join(LOG_DIR, "judge_calls.jsonl")


def ensure_log_dir():
    """Create the log directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)


def log_evaluation(
    question: str,
    groq_answer: str,
    gemini_answer: str,
    ab_winner: str,
    ba_winner: str,
    position_flip: bool,
    position_bias_detected: bool,
    scores: Dict[str, Any],
    reason: str,
    token_usage: Dict[str, int],
    latency_ms: int,
) -> None:
    """
    Appends a single evaluation log entry as one JSON line.
    Uses APPEND mode — never overwrites previous entries.
    """
    ensure_log_dir()

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "groq_answer": groq_answer,
        "gemini_answer": gemini_answer,
        "winner": ab_winner,
        "ab_winner": ab_winner,
        "ba_winner": ba_winner,
        "position_flip": position_flip,
        "position_bias_detected": position_bias_detected,
        "scores": scores,
        "reason": reason,
        "token_usage": token_usage,
        "latency_ms": latency_ms,
    }

    with open(EVALUATIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def log_judge_call(
    question: str,
    judge_model: str,
    system_prompt: str,
    user_prompt: str,
    raw_gpt_response: str,
    parsed_verdict: Dict[str, Any],
    token_usage: int,
    parsing_status: str = "success",
) -> None:
    """
    Auditable log of every GPT judge call.
    Appends to logs/judge_calls.jsonl with full prompts, raw response, parsed verdict.
    Never logs secrets or API keys.
    """
    ensure_log_dir()

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": judge_model,
        "question": question,
        "judge_prompts": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        "raw_gpt_response": raw_gpt_response,
        "parsed_verdict": parsed_verdict,
        "parsing_status": parsing_status,
        "token_usage": token_usage,
    }

    with open(JUDGE_CALLS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
