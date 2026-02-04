# AI Blog Evaluation System - Backend

A comprehensive AI-powered blog evaluation system using **LangGraph** for workflow orchestration and **FastAPI** for API endpoints. The system evaluates blogs across 8 quality metrics using 3 specialized AI agents.

## Features

- 🤖 **AI-Powered Evaluation**: Uses OpenAI models for intelligent content analysis
- 📊 **8 Quality Metrics**: Completeness, Relevance, Clarity, Factual Accuracy, Instruction Following, Style & Tone, Context Awareness, Safety
- 🔍 **AI Detection**: Identifies likelihood of AI-generated content
- 🌐 **Reference Validation**: Validates and extracts claims from reference URLs
- ⚡ **Async Processing**: Support for background job processing
- 📈 **Detailed Insights**: Actionable improvement suggestions with priorities

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
├─────────────────────────────────────────────────────────────────┤
│                      LangGraph Workflow                          │
│  ┌──────────────┐                                               │
│  │ Input        │                                               │
│  │ Validator    │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│    ┌────┴────┬────────────┬────────────┐                        │
│    ▼         ▼            ▼            ▼                        │
│ ┌──────┐ ┌──────┐   ┌──────────┐ ┌──────────┐                  │
│ │Agent │ │Agent │   │  Agent   │ │Reference │    (Parallel)     │
│ │  1   │ │  2   │   │    3     │ │Validator │                   │
│ └──┬───┘ └──┬───┘   └────┬─────┘ └────┬─────┘                   │
│    │        │            │            │                          │
│    └────────┴────────────┴────────────┘                          │
│                     │                                            │
│              ┌──────▼──────┐                                     │
│              │  Aggregator │                                     │
│              │  & Insights │                                     │
│              └──────┬──────┘                                     │
│                     │                                            │
│              ┌──────▼──────┐                                     │
│              │   Report    │                                     │
│              │  Compiler   │                                     │
│              └─────────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **FastAPI** - Modern async web framework
- **LangGraph** - Workflow orchestration
- **LangChain** - LLM integration
- **OpenAI** - LLM provider
- **PostgreSQL** - Database (optional)
- **Redis** - Caching (optional)
- **Pydantic** - Data validation

## Quick Start

### 1. Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required configuration:
```env
# Choose your LLM provider
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

### 3. Run the Server

```bash
# Development mode
uvicorn app.main:app --reload --port 8000

# Or run directly
python -m app.main
```

### 4. Access the API

- API Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Health Check: http://localhost:8000/api/v1/health

## API Endpoints

### POST /api/v1/evaluate
Full blog evaluation across all 8 metrics.

```bash
curl -X POST "http://localhost:8000/api/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "blogText": "Your blog content here...",
    "promptText": "Original writing prompt",
    "toneOfVoice": {
      "type": "text",
      "content": "Professional and educational"
    },
    "references": [],
    "targetAudience": "Python developers"
  }'
```

### POST /api/v1/evaluate (Async Mode)
Submit for background processing.

```bash
curl -X POST "http://localhost:8000/api/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "blogText": "...",
    "evaluationOptions": {"async_mode": true}
  }'
# Returns: {"jobId": "uuid", "status": "processing"}
```

### GET /api/v1/evaluate/{job_id}
Check async job status.

### POST /api/v1/evaluate/quick
Rapid assessment (3 core metrics only).

### POST /api/v1/validate/references
Standalone reference URL validation.

### GET /api/v1/metrics/definitions
Get metric definitions and scoring bands.

### GET /api/v1/health
Health check endpoint.

## Evaluation Metrics

| Metric | Weight | Target | Description |
|--------|--------|--------|-------------|
| Completeness | 20% | 63 | Topic coverage and depth |
| Relevance | 11.43% | 81 | Content alignment with title |
| Clarity | 11.43% | 89 | Readability and structure |
| Factual Accuracy | 11.43% | 87 | Claim verification |
| Instruction Following | 11.43% | 93 | Guideline adherence |
| Style & Tone | 11.43% | 90 | Voice consistency |
| Context Awareness | 11.43% | 83 | Audience appropriateness |
| Safety | 11.43% | 95 | Harmful content detection |

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration
│   ├── models/
│   │   ├── requests.py      # Request models
│   │   └── responses.py     # Response models
│   ├── agents/
│   │   ├── base_agent.py    # Base agent class
│   │   ├── content_quality_agent.py
│   │   ├── style_compliance_agent.py
│   │   └── safety_detection_agent.py
│   ├── workflows/
│   │   └── evaluation_workflow.py  # LangGraph workflow
│   ├── services/
│   │   ├── reference_validator.py
│   │   ├── aggregator.py
│   │   └── insight_generator.py
│   ├── utils/
│   │   ├── text_analysis.py
│   │   └── prompts.py
│   ├── database/
│   │   ├── models.py
│   │   └── session.py
│   └── api/
│       └── routes.py
├── tests/
│   ├── test_agents.py
│   ├── test_workflow.py
│   └── test_api.py
├── requirements.txt
├── .env.example
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_agents.py -v
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | openai | LLM provider |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OPENAI_MODEL` | gpt-4-turbo-preview | OpenAI model |
| `DATABASE_URL` | - | PostgreSQL connection URL |
| `REDIS_URL` | - | Redis connection URL |
| `DEBUG` | true | Enable debug mode |
| `CORS_ORIGINS` | [...] | Allowed CORS origins |

### Metric Weights

Weights can be customized via environment variables:
```env
WEIGHT_COMPLETENESS=0.20
WEIGHT_RELEVANCE=0.1143
# ... etc
```

## Development

### Code Formatting

```bash
black app/
isort app/
flake8 app/
```

### Database Migrations (if using PostgreSQL)

```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

## Error Handling

The API handles errors gracefully:

- **400**: Invalid request format
- **422**: Validation error (content too short, etc.)
- **500**: Internal server error

Agent failures don't crash the workflow - fallback scores (50) are used with error flags.

## Performance Considerations

- **Parallel Agent Execution**: All 3 agents run concurrently
- **Reference Caching**: URLs are cached for 24 hours
- **Async Processing**: Long evaluations can run in background
- **Connection Pooling**: Database connections are pooled

## License

MIT License - see LICENSE file for details.
