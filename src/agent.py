import json
import logging
import os
import re
from typing import Any

import openai
from pydantic import BaseModel, HttpUrl, ValidationError
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qa_evaluator")


# Judge model configurations - all use OpenAI-compatible API
JUDGE_MODELS = {
    # OpenAI models
    "gpt-4o": {
        "base_url": None,  # Uses default OpenAI endpoint
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-4o-mini": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    # Anthropic models (via OpenAI-compatible endpoint)
    "claude-sonnet-4-5": {
        "base_url": "https://api.anthropic.com/v1/",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    # Google models (via OpenAI-compatible endpoint)
    "gemini-2.5-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "gemini-2.0-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GOOGLE_API_KEY",
    },
}

DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"


def parse_evaluation_json(eval_text: str) -> dict:
    """Parse JSON evaluation response, handling potential formatting issues."""
    try:
        # Try to find JSON content between curly braces
        json_match = re.search(r'\{.*\}', eval_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            # Fallback - try to parse the whole text
            return json.loads(eval_text)
    except (json.JSONDecodeError, AttributeError):
        # Return a default structure if parsing fails
        return {
            "architecture_reasoning": {"score": 0, "feedback": "Parse error"},
            "reasoning_consistency": {"score": 0, "feedback": "Parse error"},
            "code_understanding_tier": {"tier": "unknown", "score": 0, "feedback": "Parse error"},
            "grounding": {"score": 0, "feedback": "Parse error"}
        }


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL (expects "qa_agent")
    config: dict[str, Any]  # expects "question", optional "reference_answer", and "repo_url"


class ScoreWithFeedback(BaseModel):
    """A score with detailed feedback."""
    score: int  # 0-5
    feedback: str


class CodeUnderstandingScore(BaseModel):
    """Code understanding evaluation with tier classification."""
    tier: str  # performance-related, runtime-related, inter-module, architectural
    score: int  # 0-5
    feedback: str


class AnswerScore(BaseModel):
    """Evaluation scores for a single answer."""
    architecture_reasoning: ScoreWithFeedback
    reasoning_consistency: ScoreWithFeedback
    code_understanding_tier: CodeUnderstandingScore
    grounding: ScoreWithFeedback


class EvalResult(BaseModel):
    """Evaluation result for a single question."""
    question: str
    reference_answer: str | None
    agent_answer: str
    repo_url: str
    scores: AnswerScore
    total_score: float  # average of 4 scores


class Agent:
    required_roles: list[str] = ["codewalk-qa-agent"]
    required_config_keys: list[str] = ["question", "repo_url"]

    def __init__(self):
        self.messenger = Messenger()

    def _get_judge_client(self, model_name: str) -> tuple[openai.OpenAI, str]:
        """Get OpenAI-compatible client for the specified judge model."""
        if model_name not in JUDGE_MODELS:
            logger.warning(f"Unknown model {model_name}, using default {DEFAULT_JUDGE_MODEL}")
            model_name = DEFAULT_JUDGE_MODEL

        config = JUDGE_MODELS[model_name]
        api_key = os.getenv(config["api_key_env"])

        if not api_key:
            raise ValueError(f"Missing API key: {config['api_key_env']} environment variable not set")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=config.get("base_url"),
        )

        return client, model_name

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        if not request.config.get("question"):
            return False, "Empty question provided"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        question = request.config["question"]
        reference_answer = request.config.get("reference_answer")
        repo_url = request.config["repo_url"]
        judge_model = request.config.get("judge_model", DEFAULT_JUDGE_MODEL)
        qa_agent_url = str(request.participants["codewalk-qa-agent"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Sending question to Q&A agent: {question[:100]}...")
        )

        # Send question to Q&A agent
        try:
            agent_answer = await self.messenger.talk_to_agent(
                question, qa_agent_url, new_conversation=True
            )
        except Exception as e:
            logger.error(f"Error getting answer from Q&A agent: {e}")
            await updater.reject(new_agent_text_message(f"Failed to get answer from Q&A agent: {e}"))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Received answer. Evaluating with {judge_model}...")
        )

        # Evaluate the answer
        scores = await self.evaluate_answer(question, reference_answer, agent_answer, repo_url, judge_model)

        total_score = (
            scores.architecture_reasoning.score +
            scores.reasoning_consistency.score +
            scores.code_understanding_tier.score +
            scores.grounding.score
        ) / 4.0

        eval_result = EvalResult(
            question=question,
            reference_answer=reference_answer,
            agent_answer=agent_answer,
            repo_url=repo_url,
            scores=scores,
            total_score=total_score,
        )

        logger.info(f"Evaluation complete. Total score: {total_score:.2f}/5")

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=f"Evaluation complete. Score: {total_score:.2f}/5")),
                Part(root=DataPart(data=eval_result.model_dump())),
            ],
            name="Result",
        )

    async def evaluate_answer(
        self,
        question: str,
        reference_answer: str | None,
        agent_answer: str,
        repo_url: str,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> AnswerScore:
        """Evaluate a Q&A response using LLM judge."""

        reference_section = ""
        if reference_answer:
            reference_section = f"""
**Reference Answer:** {reference_answer}
Use this reference answer as a guide for evaluating grounding and accuracy."""
        else:
            reference_section = f"""
**Reference Answer:** Not provided.
Use your knowledge of the {repo_url} open source repository to evaluate grounding and accuracy."""

        evaluation_prompt = f"""
You are a senior software engineer with 10 years of experience in software development.
The question and answer pairs are designed to help a software engineer ramp up on the codebase for {repo_url}.
The answers should follow the "life of X" style format (if the question is about understanding how something works and flows through the system, e.g., how does request processing work) so that it is helpful for an engineer to meaningfully contribute to the codebase.
Please evaluate the following response based on accuracy, coherence, reasoning consistency, and grounding:

**Question:** {question}
{reference_section}

**Model Response:** {agent_answer}

**Evaluation Criteria:**
1. **Architecture-Level Reasoning**: Does the response provide clear reasoning about the system's design, modules, or architecture? (Score 0-5)
2. **Reasoning Consistency**: Is the reasoning consistent? Does it follow a logical and coherent flow? (Score 0-5)
3. **Code Understanding Tier**: Categorize the question into one of the following tiers: performance-related, runtime-related, inter-module, or architectural. How well does the model understand the question within the given code understanding tier? (Score 0-5)
4. **Grounding Score**: How factual and accurate is the response? If a reference answer is provided, evaluate alignment with it. Otherwise, use your knowledge of the {repo_url} repository to assess factual accuracy. (Score 0-5)

Provide a detailed evaluation based on these criteria, and include the feedback and justification for each score. Be very strict in your evaluation. A high score needs to be backed by strong justification. Give your answer in the following JSON format (note: all scores should be integers, not strings):

{{
    "architecture_reasoning": {{
        "score": <int>,
        "feedback": "Detailed feedback about architecture reasoning"
    }},
    "reasoning_consistency": {{
        "score": <int>,
        "feedback": "Feedback about reasoning consistency"
    }},
    "code_understanding_tier": {{
        "tier": "tier name",
        "score": <int>,
        "feedback": "Feedback about code understanding"
    }},
    "grounding": {{
        "score": <int>,
        "feedback": "Feedback about grounding"
    }}
}}
"""

        system_message = "You are a senior software engineer with 10 years of experience in software development."

        # Get OpenAI-compatible client for the judge model
        client, resolved_model = self._get_judge_client(judge_model)

        logger.info(f"Evaluating with judge model: {resolved_model}")

        # Call the API using OpenAI-compatible format
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": evaluation_prompt}
            ],
        )

        # Parse JSON from response text
        response_text = response.choices[0].message.content.strip()
        parsed = parse_evaluation_json(response_text)

        # Convert to Pydantic model
        return AnswerScore(
            architecture_reasoning=ScoreWithFeedback(
                score=int(parsed["architecture_reasoning"]["score"]),
                feedback=parsed["architecture_reasoning"]["feedback"],
            ),
            reasoning_consistency=ScoreWithFeedback(
                score=int(parsed["reasoning_consistency"]["score"]),
                feedback=parsed["reasoning_consistency"]["feedback"],
            ),
            code_understanding_tier=CodeUnderstandingScore(
                tier=parsed["code_understanding_tier"]["tier"],
                score=int(parsed["code_understanding_tier"]["score"]),
                feedback=parsed["code_understanding_tier"]["feedback"],
            ),
            grounding=ScoreWithFeedback(
                score=int(parsed["grounding"]["score"]),
                feedback=parsed["grounding"]["feedback"],
            ),
        )