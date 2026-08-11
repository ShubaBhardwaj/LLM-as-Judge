import json
import asyncio
import time
import os
from typing import Optional, Dict, Any

from app.schemas import (
    EvaluationResponse,
    JudgeResult,
    ModelScores,
    CriterionScores,
    AnswersDict,
    TokenUsage,
    BiasCheck,
    BiasPassResult,
    QuestionItem,
)
from app.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from app.llm import LLMService
from app.logger import log_evaluation
from app.metrics import compute_suite_metrics


class LLMJudge:
    """Orchestrates Groq Generator + Gemini Generator with GPT Judge evaluation."""

    def __init__(self):
        self.llm_service = LLMService()

    # ------------------------------------------------------------------
    # Internal: Judge pre-generated answers in a given order
    # ------------------------------------------------------------------

    async def _judge_answers(
        self,
        question: str,
        option_a_content: str,
        option_b_content: str,
        option_a_label: str,
        option_b_label: str,
        reference_answer: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> tuple:
        """
        Sends two answers to GPT judge in the given order.
        Returns (mapped_winner, groq_scores, gemini_scores, reason, raw_text, judge_tokens).
        """
        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer or "N/A",
            option_a=option_a_content,
            option_b=option_b_content,
        )

        sys_prompt = system_prompt or JUDGE_SYSTEM_PROMPT

        raw_output, raw_text, judge_tokens = await self.llm_service.evaluate_with_gpt_judge(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            question=question,
        )

        # Extract scores
        scores_a = raw_output.scores.get("option_a") or raw_output.scores.get("Option A")
        scores_b = raw_output.scores.get("option_b") or raw_output.scores.get("Option B")

        default_score = CriterionScores(
            correctness=3.0, faithfulness=3.0, completeness=3.0,
            instruction_following=3.0, tone_safety=3.0, overall=3.0,
        )

        # Map back to model names based on position labels
        if option_a_label == "groq":
            groq_scores = CriterionScores(**scores_a.model_dump()) if scores_a else default_score
            gemini_scores = CriterionScores(**scores_b.model_dump()) if scores_b else default_score
            winner = {"option_a": "groq", "option_b": "gemini"}.get(raw_output.winner, "tie")
        else:
            groq_scores = CriterionScores(**scores_b.model_dump()) if scores_b else default_score
            gemini_scores = CriterionScores(**scores_a.model_dump()) if scores_a else default_score
            winner = {"option_a": "gemini", "option_b": "groq"}.get(raw_output.winner, "tie")

        return winner, groq_scores, gemini_scores, raw_output.reason, raw_text, judge_tokens

    # ------------------------------------------------------------------
    # Core: Full evaluation with inline bias check
    # ------------------------------------------------------------------

    async def evaluate_question(
        self,
        question: str,
        reference_answer: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> EvaluationResponse:
        """
        Complete evaluation pipeline for a single question:
        1. Generate answers from Groq + Gemini in parallel
        2. Judge A/B (Groq=A, Gemini=B)
        3. Judge B/A (Gemini=A, Groq=B) — same answers, swapped positions
        4. Detect position bias
        5. Log everything
        6. Return unified response
        """
        start_time = time.time()

        # Step 1: Generate answers in parallel
        (groq_answer, groq_tokens), (gemini_answer, gemini_tokens) = await asyncio.gather(
            self.llm_service.generate_groq_answer(question),
            self.llm_service.generate_gemini_answer(question),
        )

        # Step 2: Judge A/B (Groq = Option A, Gemini = Option B)
        ab_winner, ab_groq_scores, ab_gemini_scores, ab_reason, _, ab_judge_tokens = (
            await self._judge_answers(
                question=question,
                option_a_content=groq_answer,
                option_b_content=gemini_answer,
                option_a_label="groq",
                option_b_label="gemini",
                reference_answer=reference_answer,
                system_prompt=system_prompt,
            )
        )

        # Step 3: Judge B/A — SAME answers, swapped positions (no regeneration)
        ba_winner, _, _, _, _, ba_judge_tokens = (
            await self._judge_answers(
                question=question,
                option_a_content=gemini_answer,
                option_b_content=groq_answer,
                option_a_label="gemini",
                option_b_label="groq",
                reference_answer=reference_answer,
                system_prompt=system_prompt,
            )
        )

        # Step 4: Position bias detection
        position_flip = ab_winner != ba_winner
        position_bias_detected = position_flip

        bias_check = BiasCheck(
            ab=BiasPassResult(A="groq", B="gemini", winner=ab_winner),
            ba=BiasPassResult(A="gemini", B="groq", winner=ba_winner),
            position_flip=position_flip,
            position_bias_detected=position_bias_detected,
        )

        # Use A/B pass as the primary evaluation result
        evaluation = JudgeResult(
            winner=ab_winner,
            scores=ModelScores(groq=ab_groq_scores, gemini=ab_gemini_scores),
            reason=ab_reason,
        )

        total_judge_tokens = ab_judge_tokens + ba_judge_tokens
        token_usage = TokenUsage(
            groq_tokens=groq_tokens,
            gemini_tokens=gemini_tokens,
            judge_tokens=total_judge_tokens,
            total_tokens=groq_tokens + gemini_tokens + total_judge_tokens,
        )

        latency = round(time.time() - start_time, 3)
        latency_ms = int(latency * 1000)

        # Step 5: Log to JSONL
        try:
            log_evaluation(
                question=question,
                groq_answer=groq_answer,
                gemini_answer=gemini_answer,
                ab_winner=ab_winner,
                ba_winner=ba_winner,
                position_flip=position_flip,
                position_bias_detected=position_bias_detected,
                scores=evaluation.scores.model_dump(),
                reason=ab_reason,
                token_usage=token_usage.model_dump(),
                latency_ms=latency_ms,
            )
        except Exception:
            pass

        return EvaluationResponse(
            question=question,
            answers=AnswersDict(groq=groq_answer, gemini=gemini_answer),
            evaluation=evaluation,
            bias_check=bias_check,
            token_usage=token_usage,
            latency_seconds=latency,
        )

    # ------------------------------------------------------------------
    # Batch: Internal function — run_batch_evaluation()
    # ------------------------------------------------------------------

    async def run_batch_evaluation(
        self,
        questions_path: str = "data/questions.json",
        report_path: str = "results/report.json",
        cases_path: str = "results/cases.json",
        bias_report_path: str = "results/bias_results.json",
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Internal batch evaluation function — NOT exposed as an API endpoint.
        Loads questions from JSON, runs full pipeline + bias test per question,
        computes aggregate metrics & score clustering, saves results files.

        Usage:
            python -m app.batch
        """
        start_time = time.time()

        with open(questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = [QuestionItem(**item) for item in data]

        cases = []

        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Evaluating: {q.question[:60]}...")
            result = await self.evaluate_question(
                q.question, q.reference_answer, system_prompt=system_prompt
            )

            groq_len = len(result.answers.groq)
            gemini_len = len(result.answers.gemini)

            cases.append({
                "question_id": q.id,
                "category": q.category,
                "question": q.question,
                "groq_answer": result.answers.groq,
                "gemini_answer": result.answers.gemini,
                "groq_char_length": groq_len,
                "gemini_char_length": gemini_len,
                "winner": result.evaluation.winner,
                "reason": result.evaluation.reason,
                "groq_scores": result.evaluation.scores.groq.model_dump(),
                "gemini_scores": result.evaluation.scores.gemini.model_dump(),
                "ab_winner": result.bias_check.ab.winner,
                "ba_winner": result.bias_check.ba.winner,
                "position_flipped": result.bias_check.position_flip,
                "position_bias_detected": result.bias_check.position_bias_detected,
                "token_usage": result.token_usage.model_dump(),
                "total_tokens": result.token_usage.total_tokens,
                "latency_ms": int(result.latency_seconds * 1000),
            })
            await asyncio.sleep(2.0)

        # Compute aggregate metrics
        report = compute_suite_metrics(cases)
        total_latency = round(time.time() - start_time, 3)
        report["total_latency_seconds"] = total_latency

        # Extract bias summary
        bias_summary = {
            "total_questions": report["total_questions"],
            "position_bias": report["position_bias"],
            "score_clustering": report["score_clustering"],
            "verbosity_length": report["verbosity_length"],
        }

        # Save report.json (aggregate metrics)
        try:
            os.makedirs("results", exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved: {report_path}")
        except Exception as e:
            print(f"Error saving report: {e}")

        # Save cases.json (per-question details)
        try:
            with open(cases_path, "w", encoding="utf-8") as f:
                json.dump(cases, f, indent=2, ensure_ascii=False)
            print(f"Cases saved: {cases_path}")
        except Exception as e:
            print(f"Error saving cases: {e}")

        # Save bias_results.json
        try:
            with open(bias_report_path, "w", encoding="utf-8") as f:
                json.dump(bias_summary, f, indent=2)
            print(f"Bias summary saved: {bias_report_path}")
        except Exception as e:
            print(f"Error saving bias summary: {e}")

        return report
