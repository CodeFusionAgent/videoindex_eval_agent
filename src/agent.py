import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import openai
import yaml
from pydantic import BaseModel, HttpUrl, ValidationError
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("videoindex_evaluator")

EVAL_DATA_DIR = os.getenv("EVAL_DATA_DIR", "data")

# Judge model configurations - all use OpenAI-compatible API
JUDGE_MODELS = {
    "gpt-4o": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-4o-mini": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "claude-sonnet-4-5": {
        "base_url": "https://api.anthropic.com/v1/",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
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


def load_eval_data(data_dir: str) -> list[dict]:
    """Load evaluation data with correct answers from YAML files."""
    eval_data = []
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return eval_data
    for yaml_file in data_path.glob("*.yaml"):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                if data and "questions" in data:
                    num_questions = len(data["questions"])
                    eval_data.extend(data["questions"])
                    logger.info(f"Loaded {num_questions} questions from {yaml_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {yaml_file.name}: {e}")
    logger.info(f"Total evaluation questions loaded: {len(eval_data)}")
    return eval_data


def find_correct_answer(question: str, eval_data: list[dict]) -> dict | None:
    """Find the correct answer for a question."""
    question_lower = question.lower().strip()
    for qa in eval_data:
        if qa.get("question", "").lower().strip() == question_lower:
            return qa
    return None


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class EvalResult(BaseModel):
    """Evaluation result for a video Q&A question."""
    question: str
    episode: str
    clip: str
    correct_answer: str
    agent_answer: str
    similarity_score: float  # 0.0 to 1.0 based on semantic similarity
    feedback: str


class Agent:
    required_roles: list[str] = ["videoindex-qa-agent"]
    required_config_keys: list[str] = ["question"]

    def __init__(self):
        self.messenger = Messenger()
        self.eval_data = load_eval_data(EVAL_DATA_DIR)

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

    async def evaluate_semantic_similarity(
        self,
        question: str,
        correct_answer: str,
        agent_answer: str,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> tuple[float, str]:
        """Evaluate semantic similarity between agent answer and correct answer using LLM."""

        evaluation_prompt = f"""You are evaluating whether two answers to a question about a video are semantically equivalent.

**Question:** {question}

**Correct Answer:** {correct_answer}

**Agent's Answer:** {agent_answer}

Evaluate the semantic similarity between the correct answer and the agent's answer. Consider:
- Do they convey the same meaning?
- Are they referring to the same concept/event/reasoning?
- Minor wording differences are acceptable if the meaning is the same.

Respond with a JSON object:
{{
    "score": <float between 0.0 and 1.0>,
    "feedback": "<brief explanation of your scoring>"
}}

Scoring guide:
- 1.0: Semantically identical or equivalent meaning
- 0.7-0.9: Very similar, minor differences that don't change the core meaning
- 0.4-0.6: Partially similar, some overlap but missing key elements
- 0.1-0.3: Slightly related but mostly different
- 0.0: Completely different or contradictory"""

        client, resolved_model = self._get_judge_client(judge_model)

        logger.info(f"Evaluating semantic similarity with judge model: {resolved_model}")

        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator of semantic similarity between answers."},
                    {"role": "user", "content": evaluation_prompt}
                ],
            )

            response_text = response.choices[0].message.content.strip()

            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                score = float(parsed.get("score", 0.0))
                feedback = parsed.get("feedback", "No feedback provided")
                # Clamp score between 0 and 1
                score = max(0.0, min(1.0, score))
                return score, feedback
            else:
                logger.error(f"Could not parse JSON from response: {response_text}")
                return 0.0, "Failed to parse evaluation response"

        except Exception as e:
            logger.error(f"Error during semantic similarity evaluation: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

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
        judge_model = request.config.get("judge_model", DEFAULT_JUDGE_MODEL)
        qa_agent_url = str(request.participants["videoindex-qa-agent"])

        # Find the correct answer from our evaluation data
        qa_entry = find_correct_answer(question, self.eval_data)
        if not qa_entry:
            await updater.reject(new_agent_text_message(f"Question not found in evaluation dataset: {question[:100]}..."))
            return

        correct_answer = qa_entry["correct_answer"]
        episode = qa_entry.get("episode", "unknown")
        clip = qa_entry.get("clip", "unknown")

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
            new_agent_text_message(f"Received answer. Evaluating semantic similarity with {judge_model}...")
        )

        # Evaluate semantic similarity using LLM
        score, feedback = await self.evaluate_semantic_similarity(
            question, correct_answer, agent_answer, judge_model
        )

        eval_result = EvalResult(
            question=question,
            episode=episode,
            clip=clip,
            correct_answer=correct_answer,
            agent_answer=agent_answer,
            similarity_score=score,
            feedback=feedback,
        )

        logger.info(f"Evaluation complete. Similarity score: {score:.2f}")

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=f"Evaluation complete. Similarity score: {score:.2f}/1.0")),
                Part(root=DataPart(data=eval_result.model_dump())),
            ],
            name="Result",
        )
