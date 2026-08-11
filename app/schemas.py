from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal


# --- Health ---

class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Status of the application service")


# --- Score Models ---

class CriterionScores(BaseModel):
    """Scores for a single model across all 5 evaluation criteria (1-5) + overall."""
    correctness: float = Field(..., ge=1, le=5, description="Factual accuracy (1-5)")
    faithfulness: float = Field(..., ge=1, le=5, description="Absence of hallucinations (1-5)")
    completeness: float = Field(..., ge=1, le=5, description="Thoroughness (1-5)")
    instruction_following: float = Field(..., ge=1, le=5, description="Adherence to constraints (1-5)")
    tone_safety: float = Field(..., ge=1, le=5, description="Professional & safe tone (1-5)")
    overall: float = Field(..., ge=1, le=5, description="Overall weighted score (1-5)")


class ModelScores(BaseModel):
    """Scores for both candidate models."""
    groq: CriterionScores
    gemini: CriterionScores


# --- Raw Judge Output (internal, from GPT) ---

class RawOptionScores(BaseModel):
    correctness: float = Field(default=3.0, ge=1, le=5)
    faithfulness: float = Field(default=3.0, ge=1, le=5)
    completeness: float = Field(default=3.0, ge=1, le=5)
    instruction_following: float = Field(default=3.0, ge=1, le=5)
    tone_safety: float = Field(default=3.0, ge=1, le=5)
    overall: float = Field(default=3.0, ge=1, le=5)


class RawJudgeOutput(BaseModel):
    """Raw structured output from GPT judge (uses option_a/option_b)."""
    winner: Literal["option_a", "option_b", "tie"] = Field(default="tie")
    scores: Dict[str, RawOptionScores] = Field(default_factory=dict)
    reason: str = Field(default="Evaluation completed.")


# --- Judge Result (mapped to groq/gemini) ---

class JudgeResult(BaseModel):
    """Evaluation result with winner mapped to model names."""
    winner: Literal["groq", "gemini", "tie"]
    scores: ModelScores
    reason: str


class AnswersDict(BaseModel):
    """Generated answers from both candidate models."""
    groq: str
    gemini: str


class TokenUsage(BaseModel):
    """Token usage tracking for cost analysis."""
    groq_tokens: int = 0
    gemini_tokens: int = 0
    judge_tokens: int = 0
    total_tokens: int = 0


# --- Bias Check (inline within /evaluate response) ---

class BiasPassResult(BaseModel):
    """Result of a single judging pass (A/B or B/A)."""
    A: str = Field(..., description="Which model was presented as Option A")
    B: str = Field(..., description="Which model was presented as Option B")
    winner: Literal["groq", "gemini", "tie"]


class BiasCheck(BaseModel):
    """Position bias analysis: A/B vs B/A comparison."""
    ab: BiasPassResult
    ba: BiasPassResult
    position_flip: bool
    position_bias_detected: bool


# --- API Request / Response ---

class EvaluateRequest(BaseModel):
    """Request body for POST /evaluate."""
    question: str = Field(..., min_length=1)


class EvaluationResponse(BaseModel):
    """Response body for POST /evaluate — includes evaluation + bias check."""
    question: str
    answers: AnswersDict
    evaluation: JudgeResult
    bias_check: BiasCheck
    token_usage: TokenUsage
    latency_seconds: float


# --- Internal models (used by judge orchestrator & batch) ---

class QuestionItem(BaseModel):
    """A single question in the test suite."""
    id: str
    category: Optional[str] = "general"
    question: str
    reference_answer: Optional[str] = None


class ProbeResult(BaseModel):
    """Result of a single adversarial probe."""
    probe_id: str
    bias_type: str
    question: str
    expected_winner: str
    actual_winner: str
    passed: bool
    reason: str


class ValidationReport(BaseModel):
    """Report from judge validation tests."""
    test_retest_agreement_rate: float
    test_retest_flip_rate: float
    test_retest_details: List[Dict[str, Any]]
    verbosity_bias_rate: float
    sycophancy_bias_rate: float
    adversarial_results: List[ProbeResult]
    overall_reliability: str
