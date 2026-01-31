# Codewalk Eval Agent (Green Agent)

A Green Agent that evaluates Q&A agents on their ability to help software engineers interact with a codebase, build understanding of its concepts, and contribute back. Built for the [AgentBeats](https://agentbeats.dev) platform using the [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) protocol.

**Registered Agent:** [codewalk-eval-agent on AgentBeats](https://agentbeats.dev/anamsarfraz/codewalk-eval-agent)

## Abstract

**Codewalk Q&A Evaluator** benchmarks AI agents on their ability to help software engineers interact with a codebase, build understanding of its concepts, and contribute back. Given a question about a repository (e.g., "How does request processing work in FastAPI?"), the evaluator sends it to a Q&A agent via the A2A protocol, then uses an LLM judge to score the response on four dimensions:

- **Architecture-Level Reasoning** (0-5) – Clear reasoning about system design, modules, and architecture
- **Reasoning Consistency** (0-5) – Logical, coherent flow of explanation
- **Code Understanding Tier** (0-5) – Depth of understanding from performance to architectural level
- **Grounding** (0-5) – Factual accuracy and alignment with reference answers

While currently evaluating against open-source repositories, the system supports closed-source codebases as well. The benchmark supports multiple judge models (Gemini, Claude, etc.) and is part of the broader Codewalk project, which aims to build AI that maintains deep understanding of codebases from multiple software engineering perspectives—architecture, reliability, maintainability, and beyond.

## How It Works

```
┌─────────────────┐     A2A Protocol      ┌─────────────────┐
│   Eval Agent    │ ──────────────────────▶│    Q&A Agent    │
│  (Green Agent)  │                        │ (Purple Agent)  │
│                 │◀────────────────────── │                 │
│  1. Send question                        │  2. Return answer
│  3. Evaluate with LLM judge              │
│  4. Return scores + feedback             │
└─────────────────┘                        └─────────────────┘
```

## Evaluation Flow

1. **Receive Request** - EvalRequest with Q&A agent URL, question, repo_url, optional reference_answer
2. **Query Q&A Agent** - Send question via A2A protocol
3. **Evaluate Response** - LLM judge scores the answer on 4 dimensions
4. **Return Results** - Artifact with scores, feedback, and total score (average of 4 dimensions)

## Project Structure

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling and task lifecycle
├─ agent.py       # Core evaluation logic and LLM judge
└─ messenger.py   # A2A messaging utilities for inter-agent communication
tests/
└─ test_agent.py  # A2A conformance tests
Dockerfile        # Docker configuration
pyproject.toml    # Python dependencies
```

## Supported Judge Models

| Model | Provider | API Key Env Var |
|-------|----------|-----------------|
| `gemini-2.5-flash` (default) | Google | `GOOGLE_API_KEY` |
| `gemini-2.0-flash` | Google | `GOOGLE_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |
| `claude-sonnet-4-5` | Anthropic | `ANTHROPIC_API_KEY` |

All models use the OpenAI SDK with compatible endpoints for consistency.

## Running Locally

```bash
# Install dependencies
uv sync

# Set API key
export GOOGLE_API_KEY="your-api-key"

# Run the server
uv run src/server.py
```

## Running with Docker

```bash
# Build the image
docker build -t codewalk-eval-agent .

# Run the container
docker run -p 9009:9009 -e GOOGLE_API_KEY="your-key" codewalk-eval-agent
```

## Testing

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (see above)

# Run A2A conformance tests
uv run pytest --agent-url http://localhost:9009
```

## Request Format

The agent expects an `EvalRequest` JSON with:

```json
{
  "participants": {
    "codewalk-qa-agent": "http://qa-agent-url:9010"
  },
  "config": {
    "question": "How does request processing work in FastAPI?",
    "repo_url": "https://github.com/tiangolo/fastapi",
    "judge_model": "gemini-2.5-flash",
    "reference_answer": "Optional reference for grounding evaluation"
  }
}
```

## Response Format

Returns an `EvalResult` artifact:

```json
{
  "question": "...",
  "agent_answer": "...",
  "repo_url": "...",
  "total_score": 4.25,
  "scores": {
    "architecture_reasoning": {"score": 4, "feedback": "..."},
    "reasoning_consistency": {"score": 5, "feedback": "..."},
    "code_understanding_tier": {"tier": "architectural", "score": 4, "feedback": "..."},
    "grounding": {"score": 4, "feedback": "..."}
  }
}
```

## Publishing

Push to `main` to publish `latest` tag, or create a version tag (e.g., `v1.0.0`) for versioned releases:

```
ghcr.io/<your-username>/codewalk_eval_agent:latest
ghcr.io/<your-username>/codewalk_eval_agent:1.0.0
```

## Agent Registration

To register your own Green Agent on AgentBeats:

1. Follow the [AgentBeats Tutorial](https://docs.agentbeats.dev/tutorial/) for step-by-step setup
2. Build and publish your Docker image to a container registry
3. Register the agent on the [AgentBeats platform](https://agentbeats.dev)

## Related Repositories

- [codewalk_qa_agent](https://github.com/CodeFusionAgent/codewalk_qa_agent) - Baseline Purple Agent for Q&A
- [leaderboard_codewalk](https://github.com/CodeFusionAgent/leaderboard_codewalk) - Leaderboard and evaluation scenarios

## License

MIT
