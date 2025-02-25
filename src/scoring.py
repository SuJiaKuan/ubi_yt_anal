from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pydantic import Field

from src.model import create_openai_llm
from src.shape import ScoringResult

_SUPPORT_SYSTEM_PROMPT = """\
You are an expert in text analysis. Your task is to analyze user comments (from YouTube videos) about Unconditional Basic Income (UBI) and assign a support score from 1 to 10.

## Scoring Guide
- **10 (Extremely Supportive)**: Completely supports UBI with no doubts or concerns.
- **9 (Very Supportive)**: Strongly supports UBI but may mention small concerns.
- **8 (Highly Supportive)**: Generally supports UBI but discusses some challenges.
- **7 (Moderate Support)**: Supports UBI but thinks it needs careful implementation.
- **6 (Slightly Supportive)**: Somewhat supports UBI but has major concerns.
- **5 (Neutral)**: No clear stance, raises questions or discusses both sides.
- **4 (Slightly Opposed)**: Mostly against UBI but acknowledges some positives.
- **3 (Moderate Opposition)**: Generally opposes UBI but considers rare exceptions.
- **2 (Highly Opposed)**: Strongly opposes UBI, thinks it has major flaws.
- **1 (Extremely Opposed)**: Completely rejects UBI, believes it is a disastrous policy.

## Instructions
- Read the comment carefully.
- Provide a **detailed reasoning** for your score.
- Assign a **single score (1-10)** based on the user's comment.

## Format Instructions
{format_instructions}\
"""

_SENTIMENT_SYSTEM_PROMPT = """\
You are an expert in sentiment analysis. Your task is to analyze user comments (from YouTube videos) and assign a sentiment score from 1 to 10.

## Scoring Guide
- **10 (Extremely Positive):** Highly enthusiastic, excited, overwhelmingly positive.
- **9 (Very Positive):** Strongly positive, with slight reservations.
- **8 (Highly Positive):** Positive but with a bit of rational analysis.
- **7 (Somewhat Positive):** Leans positive but includes minor criticisms.
- **6 (Slightly Positive):** Generally positive but not highly enthusiastic.
- **5 (Neutral):** No strong emotions, just factual or balanced discussion.
- **4 (Slightly Negative):** Some dissatisfaction but not strongly critical.
- **3 (Moderately Negative):** Leans negative but acknowledges some positives.
- **2 (Highly Negative):** Strong dissatisfaction and criticism.
- **1 (Extremely Negative):** Extremely negative, expressing frustration or anger.

## Instructions
- Read the comment carefully.
- Provide a **detailed reasoning** for your score.
- Assign a **single score (1-10)** based on the sentiment expressed.

## Format Instructions
{format_instructions}\
"""

_USER_PROMPT = """\
User Comment:
####
{comment_text}
####\
"""


class _ModelResponse(BaseModel):
    reasoning: str = Field(description="The reasoning behind the scoring")
    score: int = Field(description="The score")


class _LLMScorer(object):
    def __init__(
        self,
        system_prompt: str,
        model_name: str = "gpt-4o-mini-2024-07-18",
        temperature: float = 0.0,
    ):
        self._llm = create_openai_llm(model_name=model_name, temperature=temperature)
        self._parser = JsonOutputParser(pydantic_object=_ModelResponse)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", _USER_PROMPT),
            ]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._chain = self._prompt | self._llm | self._parser

    def score(self, comment_text: str) -> ScoringResult:
        chain_output = self._chain.invoke(
            {
                "comment_text": comment_text,
            }
        )

        score = chain_output.get("score")
        result = ScoringResult(
            score=score,
            detail={"reasoning": chain_output.get("reasoning")},
        )

        return result


class SupportScorer(_LLMScorer):
    def __init__(self):
        super().__init__(_SUPPORT_SYSTEM_PROMPT)


class SentimentScorer(_LLMScorer):
    def __init__(self):
        super().__init__(_SENTIMENT_SYSTEM_PROMPT)
