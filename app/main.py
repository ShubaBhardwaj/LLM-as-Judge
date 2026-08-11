from fastapi import FastAPI, HTTPException
from app.schemas import HealthResponse, EvaluateRequest, EvaluationResponse
from app.judge import LLMJudge

app = FastAPI(
    title="LLM-as-a-Judge Evaluation Pipeline",
    description="Groq Generator + Gemini Generator → GPT-4o-mini Judge with inline position bias testing.",
    version="2.0.0",
)

judge = LLMJudge()


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_question(request: EvaluateRequest):
    """
    Complete evaluation pipeline in a single request:
    1. Groq generates an answer
    2. Gemini generates an answer
    3. GPT-4o-mini judges A=Groq, B=Gemini
    4. GPT-4o-mini judges A=Gemini, B=Groq (same answers, swapped)
    5. Position bias detection
    6. Logged to logs/evaluations.jsonl
    """
    try:
        return await judge.evaluate_question(question=request.question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        error_msg = str(e)
        # Never expose API keys in error messages
        if "api_key" in error_msg.lower() or "authorization" in error_msg.lower():
            error_msg = "Authentication error with an LLM provider. Check your API keys."
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {error_msg}")
