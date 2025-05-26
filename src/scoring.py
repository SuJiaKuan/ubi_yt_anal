from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pydantic import Field

from src.model import create_openai_llm
from src.shape import ScoringResult

_SUPPORT_SYSTEM_PROMPT = """\
You are an expert in text analysis. Your task is to analyze user comments (from YouTube videos) about Unconditional Basic Income (UBI) and assign a support score from 1 to 100.

## Scoring Guide
- **86–100 (Extremely Supportive)**: Completely supports UBI with no doubts or concerns. *(e.g., 87, 93, 100)*
- **71–85 (Moderately Supportive)**: Generally supports UBI but discusses some challenges. *(e.g., 72, 76, 83)*
- **56–70 (Slightly Supportive)**: Somewhat supports UBI but has major concerns. *(e.g., 57, 64, 69)*
- **46–55 (Neutral)**: No clear stance, raises questions or discusses both sides. *(e.g., 46, 52, 55)*
- **31–45 (Slightly Opposed)**: Mostly against UBI but acknowledges some positives. *(e.g., 33, 39, 42)*
- **16–30 (Moderate Opposition)**: Generally opposes UBI but considers rare exceptions. *(e.g., 18, 21, 29)*
- **1–15 (Very Opposed)**: Strongly opposes UBI, believes it is a disastrous policy. *(e.g., 2, 8, 13)*

## Instructions
- Read the comment carefully.
- Provide a **detailed reasoning** for your score.
- Assign a **single integer score between 1 and 100** that best fits the comment. **Avoid using only round numbers (like 10, 20, 30, 50, 70, 100) or always picking numbers ending in 0 or 5.** Use any number as appropriate (e.g., 33, 41, 58, 73, 91).

## Format Instructions
{format_instructions}\
"""

_INFO_DEPTH_SYSTEM_PROMPT = """\
You are an expert in text analysis. Your task is to evaluate user comments (from YouTube videos) about Unconditional Basic Income (UBI) and assign an information depth score from 1 to 100.

## Scoring Guide
- **91–100 (Highly Detailed & Analytical)**: The comment contains multiple arguments, references data, studies, or specific cases, and deeply explores the impact of UBI. *(e.g., 92, 97, 100)*
- **81–90 (Very Detailed)**: The comment presents clear arguments and refers to examples or expert opinions but lacks specific data. *(e.g., 82, 88, 90)*
- **71–80 (Highly Informative)**: The comment presents multiple viewpoints but does not provide specific data or examples. *(e.g., 73, 76, 79)*
- **61–70 (Moderately Informative)**: The comment raises relevant points but lacks depth or detailed analysis. *(e.g., 62, 65, 68)*
- **51–60 (Basic Information)**: The comment presents a simple viewpoint but does not go further into reasoning. *(e.g., 53, 57, 60)*
- **41–50 (Slightly Informative)**: The comment states a position but does not explain it well. *(e.g., 42, 48, 50)*
- **31–40 (Very Basic Information)**: The comment makes a simple statement without elaboration. *(e.g., 33, 37, 39)*
- **21–30 (Minimal Information)**: The comment is very short and lacks substance. *(e.g., 22, 25, 28)*
- **11–20 (Very Minimal Information)**: The comment is extremely brief, often just a phrase, slogan, or emoji. *(e.g., 12, 14, 18)*
- **1–10 (No Information)**: The comment is meaningless or consists of only symbols or irrelevant text. *(e.g., 3, 7, 9)*

## Instructions
- Read the comment carefully.
- Provide a **detailed reasoning** for your score.
- Assign a **single integer score between 1 and 100** that best fits the comment. **Avoid using only round numbers (like 10, 20, 30, 50, 70, 100) or always picking numbers ending in 0 or 5.** Use any number as appropriate (e.g., 33, 41, 58, 73, 91).

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
