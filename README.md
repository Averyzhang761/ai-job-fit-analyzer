# AI Job Fit Analyzer

A Gradio workbench for screening Applied AI job descriptions with an
OpenAI-compatible model endpoint. It renders structured role signals, evidence,
risks, and a recommendation, and includes a one-click regression suite for seven
labeled JD examples.

## Features

- Structured job-fit output instead of raw JSON-only display
- Editable system prompt, temperature, and max output tokens
- JSON parsing, field validation, and deterministic business rules
- Evidence and risk presentation
- One-click field-level regression report

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure an OpenAI-compatible vLLM endpoint. Keep credentials in environment
variables and never commit them:

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

## Evaluation Boundary

The current regression set contains seven manually labeled examples. A `7/7`
result only demonstrates agreement on those saved categorical fields under the
current prompt, rules, model, and parameters. It does not establish broad
real-world accuracy or long-term model stability.
