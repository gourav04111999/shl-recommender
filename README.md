# SHL Conversational Assessment Recommender

A stateless FastAPI service that guides hiring managers from vague intent to a grounded shortlist of SHL Individual Test Solutions through multi-turn dialogue.

---

## Quick Start (Local)

```bash
# 1. Clone and install
git clone <your-repo>
cd shl-recommender
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Verify
curl http://localhost:8000/health
# → {"status":"ok"}

# 5. Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role":"user","content":"I am hiring a Java developer who works with stakeholders"}]}'
```

---

## Deployment (Render — free tier)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo.
3. Render auto-detects `render.yaml`.
4. Set the `ANTHROPIC_API_KEY` environment variable in the Render dashboard.
5. Deploy. The `/health` endpoint wakes up within 2 minutes on cold start.

Alternative free platforms: **Fly.io**, **Railway**, **Hugging Face Spaces** (Docker).

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200. Used as a readiness probe.

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring a Java developer, 4 years experience"},
    {"role": "assistant", "content": "...previous reply..."},
    {"role": "user",      "content": "Mid-level, needs stakeholder communication"}
  ]
}
```

- `messages`: full conversation history, alternating user/assistant roles.
- Maximum **8 turns** (enforced — 422 if exceeded).
- Each call has a **30 second** timeout budget.

**Response:**
```json
{
  "reply": "Here are 4 assessments for a mid-level Java developer with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)",           "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",                 "url": "https://www.shl.com/...", "test_type": "P"},
    {"name": "Verify Verbal Reasoning","url": "https://www.shl.com/...", "test_type": "A"},
    {"name": "Project Management",     "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

| Field | Type | Description |
|---|---|---|
| `reply` | `string` | Conversational response |
| `recommendations` | `array[0..10]` | Empty when clarifying/refusing. 1–10 items when recommending |
| `end_of_conversation` | `bool` | `true` only when task is complete |

**Test type codes:**

| Code | Meaning |
|---|---|
| A | Ability / Cognitive |
| K | Knowledge / Skills |
| P | Personality |
| S | Situational Judgement / Simulation |
| B | Behavioural / Motivation |

---

## Project Structure

```
shl-recommender/
├── main.py          # FastAPI app — endpoints, validation, error handling
├── agent.py         # LLM orchestration, prompt, JSON extraction, catalog guard
├── catalog.py       # Catalog loader, URL index, recommendation validator
├── scraper.py       # One-time Playwright scraper to refresh catalog.json
├── data/
│   └── catalog.json # ~60 pre-built Individual Test Solutions (Individual only)
├── tests/
│   └── test_agent.py # Behavioral probes, schema checks, Recall@10
├── Dockerfile
├── render.yaml
└── requirements.txt
```

---

## Refreshing the Catalog

The bundled `data/catalog.json` covers ~60 known Individual Test Solutions. To scrape the live SHL catalog:

```bash
pip install playwright && playwright install chromium
python scraper.py
```

The scraper uses Playwright (headless Chromium) because SHL's catalog page is JavaScript-rendered and returns 403 to plain HTTP clients. It pages through `?start=0,12,24,...` with `type=1` (Individual Test Solutions), then enriches each item with a detail-page visit.

---

## Running Tests

```bash
# Unit tests (requires running server on localhost:8000)
ANTHROPIC_API_KEY=sk-ant-... uvicorn main:app --port 8000 &
pytest tests/ -v

# Point at a deployed endpoint
TEST_BASE_URL=https://your-app.onrender.com pytest tests/ -v
```

---

## Approach Document

*2-page summary as required by the assignment.*

### Problem Decomposition

The task decomposes into four sub-problems:

1. **Catalog ingestion** — obtain and structure the SHL Individual Test Solutions catalog in a machine-readable form.
2. **Retrieval** — given a user query, surface the right catalog subset for the LLM.
3. **Agent design** — decide when to clarify, recommend, refine, compare, or refuse.
4. **Grounding + hallucination guard** — ensure every URL in the output provably exists in the catalog.

### Retrieval Strategy

**Full catalog in context** (chosen) vs. vector store:

The SHL Individual Test Solutions catalog contains ~60–120 items. At ~150 tokens per item this fits comfortably in Claude Haiku's 200 K context window (~9 K tokens for 60 items). Embedding the full catalog in the system prompt gives:

- Zero cold-start overhead (no vector DB to initialise on Render's free tier).
- Perfect recall — no retrieval misses from chunking or embedding drift.
- Simpler codebase — no FAISS/Chroma dependency.

Trade-off: if the catalog grows beyond ~500 items, a vector store (FAISS or pgvector) with semantic search becomes necessary. The `catalog.py` module is designed for that swap: replace `get_full_catalog_text()` with a top-k retrieval function and inject only the relevant subset.

### Prompt Design

The system prompt has three sections:

1. **Catalog block** — serialised as plain text (`NAME / URL / TEST_TYPE / DESCRIPTION / JOB_LEVELS`). Plain text outperforms JSON here: the LLM tokenises it more efficiently and hallucination rates drop when the format mirrors natural language.
2. **Behavioural rules** — explicit triggers for CLARIFY / RECOMMEND / REFINE / COMPARE / REFUSE states with worked examples. Rules are positive ("do X when Y") rather than negative ("don't do Z") — empirically more reliable with Claude.
3. **Output format contract** — the model is instructed to return only a JSON object, no markdown fences, no preamble. A regex-based fallback in `agent.py` strips fences and extracts the JSON block if the model wraps it anyway.

### Agent Design

The agent is **single-turn stateless**: each `/chat` call is an independent inference with the full history in context. This matches the assignment spec and avoids state management complexity.

Decision logic (encoded in the system prompt, not in code):

```
turn 1, vague query → CLARIFY (ask 1 question, empty recommendations)
turn 1, job description → RECOMMEND immediately
turn N, new constraint → REFINE existing shortlist
"difference between X and Y?" → COMPARE (grounded in catalog only)
off-topic / injection → REFUSE (empty recommendations)
```

A soft rule in the prompt ("stop clarifying after 3 turns — commit to a recommendation") ensures the conversation converges before the 8-turn cap.

### Hallucination Guard

After the LLM responds, `catalog.py:validate_recommendations()` checks every returned URL against the canonical URL set from `data/catalog.json`. Items not in that set are silently dropped. If the LLM returns a correct name but a hallucinated URL, the validator falls back to a name-indexed lookup and substitutes the real URL. This makes the guard resilient to minor URL formatting errors.

### Model Choice

**Claude Haiku** is the default (`claude-haiku-4-5`). Haiku's median response latency is ~3–5 s for this prompt size, well within the 30 s timeout. Swap to `claude-sonnet-4-6` via the `LLM_MODEL` environment variable for higher accuracy at the cost of ~2× latency.

### Evaluation

Three evaluation layers:

1. **Hard evals** — schema validation on every response (automated by the test suite + the assignment harness).
2. **Behavioral probes** — 15 small conversations with binary assertions (clarify-before-recommend, refuse-off-topic, refine-honors-edits, no-hallucinated-URLs).
3. **Recall@10** — five public traces with labeled expected shortlists. Mean Recall@10 is logged by the test suite. Current mean on public traces: ~0.60–0.70 depending on model choice.

### What Didn't Work

- **Plain HTTP scraping** of SHL's catalog: returns 403. Playwright (headless Chromium) is required.
- **Asking Claude to output a Python dict** instead of JSON: the model occasionally mixes quote styles. Pure JSON instruction is more reliable.
- **Putting the catalog in the user turn** instead of the system prompt: the model treated it as user-supplied data and was less reliable about treating it as ground truth.

### Tools Used

- Claude.ai / Claude API for development iteration and code review.
- Anthropic Python SDK for the LLM integration.
- FastAPI + Pydantic for the service layer.
- Playwright for the catalog scraper.
- Render.com for free-tier deployment.
