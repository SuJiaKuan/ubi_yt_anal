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

_INFO_DEPTH_SYSTEM_PROMPT = """\
You are an expert in text analysis. Your task is to evaluate user comments (from YouTube videos) about Unconditional Basic Income (UBI) and assign an information depth score from 1 to 10.

## Scoring Guide
- **10 (Highly Detailed & Analytical)**: The comment contains multiple arguments, references data, studies, or specific cases, and deeply explores the impact of UBI.
- **9 (Very Detailed)**: The comment presents clear arguments and refers to examples or expert opinions but lacks specific data.
- **8 (Highly Informative)**: The comment presents multiple viewpoints but does not provide specific data or examples.
- **7 (Moderately Informative)**: The comment raises relevant points but lacks depth or detailed analysis.
- **6 (Basic Information)**: The comment presents a simple viewpoint but does not go further into reasoning.
- **5 (Slightly Informative)**: The comment states a position but does not explain it well.
- **4 (Very Basic Information)**: The comment makes a simple statement without elaboration.
- **3 (Minimal Information)**: The comment is very short and lacks substance.
- **2 (Very Minimal Information)**: The comment is extremely brief, often just a phrase, slogan, or emoji.
- **1 (No Information)**: The comment is meaningless or consists of only symbols or irrelevant text.

## Instructions
- Read the comment carefully.
- Provide a **detailed reasoning** for your score.
- Assign a **single score (1-10)** based on the user's comment.

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


class InfomationDepthScorer(_LLMScorer):
    def __init__(self):
        super().__init__(_INFO_DEPTH_SYSTEM_PROMPT)
