# VideoIndex Eval Agent (Green Agent)

A Green Agent that evaluates Q&A agents on their ability to answer questions about video content. Built for the [AgentBeats](https://agentbeats.dev) platform using the [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) protocol.

## Abstract

**VideoIndex Evaluator** benchmarks AI agents on their ability to understand and answer questions about video content. We believe the way we interact with video is about to change forever—instead of passively watching recordings, you could have a conversation with them.

Today, a 10-hour lecture or a long keynote is a knowledge graveyard. The insights are there, but you can't find them when you need them. We're tackling this by building interactive AI that "watches" entire videos on your behalf, transforming long, static recordings into a conversational knowledge base.

The evaluator sends questions to a Q&A agent via the A2A protocol, then uses an LLM judge to measure **semantic similarity** between the response and the correct answer.

## How It Works

```
┌─────────────────┐     A2A Protocol      ┌─────────────────┐
│   Eval Agent    │ ──────────────────────▶│    Q&A Agent    │
│  (Green Agent)  │                        │ (Purple Agent)  │
│                 │◀────────────────────── │                 │
│  1. Send question                        │  2. Return answer
│  3. Evaluate semantic similarity         │
│  4. Return score + feedback              │
└─────────────────┘                        └─────────────────┘
```

## Evaluation Flow

1. **Receive Request** - EvalRequest with Q&A agent URL, question, and judge_model
2. **Lookup Correct Answer** - Find the correct answer from evaluation data
3. **Query Q&A Agent** - Send question via A2A protocol
4. **Evaluate Semantic Similarity** - LLM judge compares agent answer to correct answer
5. **Return Results** - Artifact with similarity score (0.0-1.0) and feedback

## Scoring Guide

- **1.0**: Semantically identical or equivalent meaning
- **0.7-0.9**: Very similar, minor differences that don't change the core meaning
- **0.4-0.6**: Partially similar, some overlap but missing key elements
- **0.1-0.3**: Slightly related but mostly different
- **0.0**: Completely different or contradictory

## Project Structure

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling and task lifecycle
├─ agent.py       # Core evaluation logic with LLM judge
└─ messenger.py   # A2A messaging utilities for inter-agent communication
data/
└─ longtvqa.yaml  # Questions with correct answers
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
| `claude-sonnet-4-5` | Anthropic | `ANTHROPIC_API_KEY` |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EVAL_DATA_DIR` | Directory containing YAML evaluation files | `data` |
| `GOOGLE_API_KEY` | API key for Gemini models | Required for Gemini |
| `ANTHROPIC_API_KEY` | API key for Claude models | Required for Claude |
| `OPENAI_API_KEY` | API key for GPT models | Required for GPT |

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
docker build -t videoindex-eval-agent .

# Run the container
docker run -p 9009:9009 -e GOOGLE_API_KEY="your-key" videoindex-eval-agent
```

## Request Format

The agent expects an `EvalRequest` JSON with:

```json
{
  "participants": {
    "videoindex-qa-agent": "http://qa-agent-url:9010"
  },
  "config": {
    "question": "Why did Raj tell himself to turn his pelvis when Penny was giving him a hug?",
    "judge_model": "gemini-2.5-flash"
  }
}
```

## Response Format

Returns an `EvalResult` artifact:

```json
{
  "question": "Why did Raj tell himself to turn his pelvis...?",
  "episode": "s01e02",
  "clip": "s01e02_seg02_clip_12",
  "correct_answer": "Raj had become excited and did not want Penny to know.",
  "agent_answer": "Raj was trying to get away from Penny.",
  "similarity_score": 0.2,
  "feedback": "The agent's answer suggests avoidance while the correct answer indicates embarrassment from physical arousal. These are fundamentally different motivations."
}
```

## Related Repositories

- [videoindex_qa_agent](https://github.com/CodeFusionAgent/videoindex_qa_agent) - Baseline Purple Agent for Q&A
- [leaderboard_videoindex](https://github.com/CodeFusionAgent/leaderboard_videoindex) - Leaderboard and evaluation scenarios

## License

MIT
