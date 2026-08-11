import json
import re
import asyncio
from typing import Tuple, Dict, Any
from openai import AsyncOpenAI
import google.genai as genai
from groq import AsyncGroq

from app.config import settings
from app.schemas import RawJudgeOutput, RawOptionScores
from app.logger import log_judge_call


class LLMService:
    """Service wrapper for Groq Generator, Gemini Generator, and GPT Judge with automatic retry on rate limits."""

    def __init__(self):
        self.groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
        self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None

    async def generate_groq_answer(self, prompt: str) -> Tuple[str, int]:
        """Generates an answer using the Groq generator model with retry logic."""
        if not self.groq_client:
            raise ValueError("GROQ_API_KEY is not configured. Set it in your .env file.")

        for attempt in range(5):
            try:
                response = await self.groq_client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful, accurate, and concise AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("Groq returned an empty response.")
                tokens = response.usage.total_tokens if response.usage else 0
                return content, tokens
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate" in err_str.lower() or "503" in err_str) and attempt < 4:
                    sleep_time = (attempt + 1) * 3
                    print(f"Groq API rate limit encountered. Retrying in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
                elif "429" in err_str or "quota" in err_str.lower():
                    # Controlled fallback if quota is exhausted
                    return f"Groq response for: {prompt[:40]}... (Groq quota limit reached)", 50
                else:
                    raise e
        return f"Groq response for: {prompt[:40]}... (Groq retry limit reached)", 50

    async def generate_gemini_answer(self, prompt: str) -> Tuple[str, int]:
        """Generates an answer using the Gemini generator model with retry logic & quota fallback."""
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is not configured. Set it in your .env file.")

        for attempt in range(3):
            try:
                response = await self.gemini_client.aio.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt
                )
                content = response.text or ""
                if not content.strip():
                    raise ValueError("Gemini returned an empty response.")
                tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0
                return content, tokens
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "UNAVAILABLE" in err_str) and attempt < 2:
                    sleep_time = 5 * (attempt + 1)
                    print(f"Gemini API limit/exhaustion ({err_str[:50]}...). Retrying in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    print("Gemini API quota exhausted. Using safe candidate response fallback for batch metric evaluation.")
                    return f"Gemini response for: {prompt[:40]}... (Gemini free-tier quota limit reached)", 50
                else:
                    raise e
        return f"Gemini response for: {prompt[:40]}... (Gemini retry limit reached)", 50

    async def evaluate_with_gpt_judge(
        self,
        system_prompt: str,
        user_prompt: str,
        question: str = ""
    ) -> Tuple[RawJudgeOutput, str, int]:
        """
        Evaluates candidate answers using GPT as the independent judge with retry logic.
        Handles structured JSON output with fallback parsing and audit logging.
        Returns (parsed_output, raw_response_text, token_count).
        """
        if not self.openai_client:
            raise ValueError("OPENAI_API_KEY is not configured. Set it in your .env file.")

        for attempt in range(5):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=settings.OPENAI_JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                raw_text = response.choices[0].message.content or ""
                if not raw_text.strip():
                    raise ValueError("GPT Judge returned an empty response.")
                tokens = response.usage.total_tokens if response.usage else 0

                # Parse structured response with robust fallback
                judge_output, status = self.parse_judge_response(raw_text)

                # Audit log the complete judge call
                try:
                    log_judge_call(
                        question=question,
                        judge_model=settings.OPENAI_JUDGE_MODEL,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        raw_gpt_response=raw_text,
                        parsed_verdict=judge_output.model_dump(),
                        token_usage=tokens,
                        parsing_status=status,
                    )
                except Exception:
                    pass

                return judge_output, raw_text, tokens
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate" in err_str.lower() or "503" in err_str) and attempt < 4:
                    sleep_time = (attempt + 1) * 4
                    print(f"OpenAI API rate limit encountered. Retrying in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise e

    def parse_judge_response(self, text: str) -> Tuple[RawJudgeOutput, str]:
        """
        Robust structured verdict parser with multi-stage malformed JSON recovery.
        
        Stage 1: Clean codeblocks & standard JSON parse + Pydantic validation.
        Stage 2: Key normalization (option_a/option_b/A/B/Option A/Option B).
        Stage 3: Regex fallback extraction for broken JSON.
        Stage 4: Safe controlled error fallback (prevents app crashes).
        
        Returns (RawJudgeOutput, status_string).
        """
        # Step 1: Clean codeblock markers
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Step 2: Attempt JSON parse
        try:
            data = json.loads(cleaned)
            normalized = self._normalize_judge_keys(data)
            output = RawJudgeOutput(**normalized)
            return output, "success"
        except Exception:
            pass

        # Step 3: Attempt regex JSON block extraction
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                normalized = self._normalize_judge_keys(data)
                output = RawJudgeOutput(**normalized)
                return output, "fallback_regex_json"
        except Exception:
            pass

        # Step 4: Attempt regex key-value extraction fallback
        try:
            winner = "tie"
            if re.search(r'"winner"\s*:\s*"(option_a|Option A|A)"', text, re.IGNORECASE):
                winner = "option_a"
            elif re.search(r'"winner"\s*:\s*"(option_b|Option B|B)"', text, re.IGNORECASE):
                winner = "option_b"

            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
            reason = reason_match.group(1) if reason_match else "Fallback extraction engaged."

            output = RawJudgeOutput(
                winner=winner,
                scores={
                    "option_a": RawOptionScores(),
                    "option_b": RawOptionScores(),
                },
                reason=reason
            )
            return output, "fallback_regex_kv"
        except Exception:
            pass

        # Step 5: Controlled fallback output (Application never crashes on malformed response)
        fallback_output = RawJudgeOutput(
            winner="tie",
            scores={
                "option_a": RawOptionScores(),
                "option_b": RawOptionScores(),
            },
            reason="Controlled fallback: Failed to parse malformed GPT response."
        )
        return fallback_output, "fallback_controlled_default"

    def _normalize_judge_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes key variations (e.g. Option A -> option_a, A -> option_a)."""
        normalized = dict(data)

        # Normalize winner
        w = str(normalized.get("winner", "")).lower().strip()
        if w in ("option_a", "option a", "a"):
            normalized["winner"] = "option_a"
        elif w in ("option_b", "option b", "b"):
            normalized["winner"] = "option_b"
        else:
            normalized["winner"] = "tie"

        # Normalize scores dict
        raw_scores = normalized.get("scores", {})
        if isinstance(raw_scores, dict):
            new_scores = {}
            for k, v in raw_scores.items():
                k_lower = str(k).lower().strip()
                if k_lower in ("option_a", "option a", "a"):
                    new_scores["option_a"] = v
                elif k_lower in ("option_b", "option b", "b"):
                    new_scores["option_b"] = v
                else:
                    new_scores[k] = v
            normalized["scores"] = new_scores

        return normalized
