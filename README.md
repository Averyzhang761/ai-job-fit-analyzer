# AI Job Fit Analyzer

A Gradio application that extracts source-backed job responsibilities, maps each
responsibility to one or more work activities, evaluates candidate constraints,
and produces a deterministic apply, skip, or human-review decision.

## Current Architecture

There is one active pipeline:

    app.py
      -> analyzer.py
         -> llm_client.py
         -> models.py

- `models.py`: Pydantic contracts for responsibilities, source constraints,
  focused location review, and the final decision.
- `llm_client.py`: OpenAI-compatible Structured Output boundary, evidence
  chunking, provider Schema normalization, and typed request/response errors.
- `analyzer.py`: extraction prompt, deterministic policy, conditional location
  reviewer, and JSONL run logging.
- `app.py`: Gradio interface and presentation only.
- `scripts/run_evaluation.py`: batch runner for the 33 saved real JDs.
- `scripts/fetch_full_job_descriptions.py`: refreshes official job text.

Earlier V2-V5 implementations were removed from the active branch after their
results were recorded in the course retrospective.

## Decision Boundary

The model does:

- extract explicit responsibilities using exact source IDs;
- attach one or more activity labels to each responsibility;
- extract stated location, remote scope, sponsorship, and citizenship or
  green-card requirements;
- perform a focused location judgment only when source facts cannot be resolved
  deterministically.

Pydantic does:

- enforce required fields, types, enums, and no extra fields;
- resolve evidence IDs back to original text;
- reject responsibility quotations absent from the JD;
- reject inconsistent final review states.

Python does:

- deduplicate and aggregate activity tags;
- derive work-authorization eligibility from explicit source statements;
- apply candidate constraints;
- produce the final recommendation.

Exact quotations prove evidence existence. They do not prove extraction
completeness or activity-label accuracy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure OpenRouter:

```bash
export OPENROUTER_API_KEY="<your-api-key>"
export OPENROUTER_MODEL="openai/gpt-4.1-mini"
```

Or configure an OpenAI-compatible RunPod endpoint:

```bash
export RUNPOD_BASE_URL="https://<pod-id>-8000.proxy.runpod.net"
export RUNPOD_API_KEY="<your-api-key>"
export RUNPOD_MODEL="Qwen/Qwen3-8B"
```

## Run

```bash
python app.py
```

```bash
python -m unittest -v
```

Focused batch smoke test:

```bash
python scripts/run_evaluation.py \
  --ids REAL-013,REAL-027 \
  --output output/evaluation-smoke.json
```

The 33 saved labels in `data/eval_jds.real.jsonl` remain provisional. A
successful request is not an accuracy pass. V6 requires responsibility-level
human ground truth before it can claim a reliable activity or recommendation
score.
