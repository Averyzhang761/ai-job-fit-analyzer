# AI Job Fit Analyzer

A Gradio workbench for screening Applied AI job descriptions with an
OpenAI-compatible model endpoint. It renders structured role signals, evidence,
risks, and a recommendation. It includes a seven-case quick regression suite
and a source-backed real-posting dataset for broader baseline evaluation.

## Features

- Structured job-fit output instead of raw JSON-only display
- Editable system prompt and max output tokens
- Fixed low-temperature configuration for reproducible classification
- Exact-quote evidence linked to each assessed field
- Deterministic fit score, recommendation, and human-review decision
- Typed request, response, contract, and business failure boundaries
- JSONL run history and field-level regression reporting

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure OpenRouter (recommended for the current baseline). Keep credentials
in environment variables and never commit them:

```bash
export OPENROUTER_API_KEY="<your-api-key>"
export OPENROUTER_MODEL="openai/gpt-4.1-mini"
```

Alternatively, configure an OpenAI-compatible vLLM endpoint:

```bash
export RUNPOD_BASE_URL="https://<pod-id>-8000.proxy.runpod.net"
export RUNPOD_API_KEY="<your-api-key>"
export RUNPOD_MODEL="Qwen/Qwen3-8B"
```

Start the app:

```bash
python app.py
```

Run local tests:

```bash
python -m unittest test_app.py
```

## Architecture

- `models.py`: Pydantic contracts for extracted facts and final results
- `llm_client.py`: OpenAI-compatible request and response boundary
- `business_rules.py`: deterministic scoring and terminal decisions
- `analysis_service.py`: shared orchestration and run logging
- `evaluation.py`: field-level regression comparison
- `app.py`: Gradio presentation and event wiring

Every successful or failed analysis is appended to
`output/run_history.jsonl`. The log records model and prompt configuration but
never stores the API key.

## Evaluation Boundary

`data/eval_jds.example.jsonl` remains the fast seven-case regression set. A
`7/7` result only demonstrates agreement on those saved fields under the
current prompt, rules, model, and parameters.

`data/eval_jds.real.jsonl` is the larger source-backed baseline. It contains 33
complete job-posting bodies fetched from the official Greenhouse and Ashby job
board APIs: 11 `apply`, 12 `skip`, and 10 `human_review` provisional labels,
without application-form fields. Refresh the bodies with:

```bash
.venv/bin/python fetch_full_job_descriptions.py
```

Postings that disappeared from their public board were replaced with current
official postings; `replaced_source_url` retains the prior URL for auditing.
The labels remain provisional and must be reviewed before they are treated as
ground truth. Running this set still does not establish broad real-world
accuracy or long-term model stability.

## V6 Architecture Candidate

The Gradio application continues to use the tested production path. V6 is kept
as an evaluation candidate and is not wired into the default UI. It extracts
exact responsibility quotations with multiple activity tags and lets Python
derive the recommendation from evidence and candidate constraints.

V6 avoids exclusive role
classification and unsupported importance estimates. It has not passed a full
ground-truth evaluation. Exact quotations prove that returned evidence exists,
but they do not prove extraction completeness or activity-label accuracy.

Earlier V3-V5 experiments were removed from the main branch after their results
were recorded in the course retrospective. Future architecture experiments
should use dedicated branches and merge back only after evaluation.
