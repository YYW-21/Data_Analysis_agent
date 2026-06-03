# DataPilot: Data Analysis Agent

DataPilot is an initial, controllable tabular data analysis workflow. It accepts a CSV/XLSX dataset, profiles the data, infers the task, runs EDA, applies deterministic preprocessing, trains several scikit-learn models, evaluates them, and generates a Markdown report. LLM usage is optional and limited to report writing.

## What This Version Does

- Upload CSV/XLSX datasets
- Generate a dataset profile
- Infer classification or regression tasks
- Run basic EDA and save charts
- Build a deterministic sklearn preprocessing pipeline
- Train baseline candidate models
- Evaluate classification/regression metrics
- Generate a Markdown report
- Optionally enhance the report with an OpenAI-compatible LLM API

## Quick Start

```bash
uv sync
cp .env.example .env
uv run uvicorn datapilot.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API Flow

1. Upload a dataset:

```bash
curl -F "file=@examples/titanic_sample.csv" http://127.0.0.1:8000/datasets/upload
```

2. Start an analysis job:

```bash
curl -X POST http://127.0.0.1:8000/jobs \
  -H "Content-Type: application/json" \
  -d "{\"dataset_id\":\"<dataset_id>\",\"user_goal\":\"predict survival\",\"target_column\":\"survived\"}"
```

3. Read the generated report:

```bash
curl http://127.0.0.1:8000/jobs/<job_id>/report
```

## LLM Configuration

LLM reporting is optional. The project uses the official `openai` Python SDK and supports OpenAI-compatible gateways via environment variables:

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
OPENAI_MODEL=gpt-4o-mini
ENABLE_LLM_REPORT=true
```

If no API key is configured, DataPilot falls back to a deterministic Markdown report.

## Project Structure

```text
src/datapilot/
  agents/          workflow orchestration and report generation
  api/             FastAPI routers
  core/            config and storage helpers
  schemas/         request/response models
  tools/           deterministic profiling, EDA, ML, plotting tools
```

## Design Principle

The first version intentionally keeps the core ML workflow deterministic. LLMs are not allowed to execute arbitrary code. This makes the system easier to debug, evaluate, and explain in interviews.

