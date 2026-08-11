# LLM-as-a-Judge Evaluation Pipeline

An automated framework for evaluating candidate LLM responses using **Groq Generator** and **Gemini Generator** with an independent **GPT-4o-mini Judge**, including position bias testing, adversarial probes, score clustering analytics, and A/B prompt comparison.

---

## 🚀 Quick Start (< 10 Minutes)

### 1. Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12
- **OS**: macOS, Linux, or Windows (WSL / PowerShell)
- **Git**: Installed

### 2. Installation
```bash
# Clone repository and navigate to root
cd llm-as-judge

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```

**Required Environment Variables (`.env`):**
```env
# Groq Generator Configuration
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# Gemini Generator Configuration
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-2.5-flash

# OpenAI Judge Configuration
OPENAI_API_KEY=sk-proj-...
OPENAI_JUDGE_MODEL=gpt-4o-mini

# Logging
LOG_LEVEL=INFO
```

---

## 🤖 AI Usage Disclosure

This project uses the following AI models:
- **Groq Generator**: `llama-3.3-70b-versatile` (Answer Candidate A)
- **Gemini Generator**: `gemini-2.5-flash` (Answer Candidate B)
- **GPT Judge**: `gpt-4o-mini` (Independent Evaluator)

---

## 🏗️ System Architecture & Execution Flow

```
                    USER QUESTION
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
           GROQ                    GEMINI
        GENERATOR                 GENERATOR
     (llama-3.3-70b)           (gemini-2.5-flash)
             │                       │
             └───────────┬───────────┘
                         ▼
                    GPT-4o-mini
                       JUDGE
                         │
                ┌────────┴────────┐
                ▼                 ▼
             Pass 1            Pass 2
           (A/B Swap)        (B/A Swap)
                │                 │
                └────────┬────────┘
                         ▼
                  POSITION BIAS
                      CHECK
                         │
                         ▼
             AUDITABLE JSONL LOGGING
```

---

## 📋 Evaluation Rubric

Candidates are evaluated across 5 criteria (1–5 scale) plus overall score:

| Criterion | Definition | Scale Guidance |
|---|---|---|
| **Correctness** | Factual accuracy and absence of logical errors | 1 = Mostly wrong, 3 = Partial, 5 = Fully correct |
| **Faithfulness** | Absence of hallucinations or ungrounded claims | 1 = Hallucinated, 3 = Minor gaps, 5 = Fully faithful |
| **Completeness** | Thoroughness in covering all prompt aspects | 1 = Incomplete, 3 = Moderate, 5 = Fully complete |
| **Instruction Following** | Strict adherence to constraints & formatting | 1 = Failed, 3 = Followed most, 5 = Perfect |
| **Tone / Safety** | Professional, safe, and objective tone | 1 = Improper/unsafe, 3 = Acceptable, 5 = Excellent |
| **Overall** | Weighted quality rating | 1.0 – 5.0 |

---

## 💻 How to Run

### 1. Start FastAPI Web Server
```bash
uvicorn app.main:app --reload --port 8000
```
- API Docs (Swagger UI): `http://localhost:8000/docs`
- Endpoints exposed:
  - `GET /health` — Basic health check
  - `POST /evaluate` — Evaluates a single question with inline position bias check & logging

#### Example Request (`POST /evaluate`):
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain what an API is in simple words."}'
```

---

### 2. Run Batch Evaluation Suite
Runs the 20-question test suite (`data/questions.json`) with double-pass position bias testing and produces aggregate metrics:

```bash
python -m app.batch
```

**Deliverables Generated:**
- `results/report.json` — Suite aggregate metrics (win rates, 5-criterion mean scores, score clustering std dev, position flip rate)
- `results/cases.json` — Detailed per-question results

---

### 3. Run Judge Validation & Adversarial Probes
Runs test-retest consistency checks (5 runs) and 6 adversarial probes (verbosity and sycophancy probes):

```bash
python -m app.validate
```

**Deliverables Generated:**
- `results/validation_results.json` — Agreement rates, flip rates, and probe results

---

### 4. Run A/B Judge Prompt Comparison
Compares Config A (Baseline Judge Prompt) vs Config B (Anti-Bias Enhanced Prompt):

```bash
python -m app.ab_comparison
```

**Deliverables Generated:**
- `results/ab_comparison.json` — Win rates, mean scores, position flip rates, and declared winning configuration

---

## 📁 Project Structure

```
llm-as-judge/
├── app/
│   ├── main.py            # FastAPI endpoints (GET /health, POST /evaluate)
│   ├── schemas.py         # Pydantic models for evaluation & bias
│   ├── llm.py             # Service wrapper for Groq, Gemini & GPT-4o-mini with fallback parsing
│   ├── judge.py           # Evaluation orchestrator & run_batch_evaluation()
│   ├── batch.py           # CLI runner for batch evaluation suite
│   ├── validate.py        # CLI runner for judge validation & adversarial probes
│   ├── ab_comparison.py   # CLI runner for A/B judge prompt comparison
│   ├── logger.py          # Auditable JSONL logging (evaluations.jsonl & judge_calls.jsonl)
│   ├── metrics.py         # Aggregate metrics, per-criterion means & score clustering
│   ├── prompts.py         # Config A (Baseline) and Config B (Anti-Bias) rubrics
│   └── config.py          # pydantic-settings environment configuration
├── data/
│   ├── questions.json     # 20 fixed test suite questions
│   └── adversarial_probes.json # 6 adversarial probe test cases
├── docs/
│   ├── architecture.md    # Architecture diagram & flow explanation
│   └── design_decisions.md# Answers to design questions & release gating guardrails
├── logs/
│   ├── evaluations.jsonl  # Request-level evaluation logs (append mode)
│   └── judge_calls.jsonl  # Raw judge prompt & response logs (auditable)
├── results/
│   ├── report.json        # Suite aggregate metrics
│   ├── cases.json         # Per-question evaluation breakdown
│   ├── bias_results.json  # Position bias & score clustering summary
│   ├── validation_results.json # Test-retest & adversarial probe results
│   └── ab_comparison.json # Config A vs Config B comparison report
├── .env.example           # Environment template with variable names
├── README.md              # Setup & run documentation
└── requirements.txt       # Dependencies
```
