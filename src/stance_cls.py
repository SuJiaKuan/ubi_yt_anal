from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pydantic import Field

from src.const import STANCE_MAJOR_CATEGORY
from src.const import STANCE_MINOR_CATEGORY
from src.model import create_openai_llm
from src.shape import StanceClassificationResult

_SYSTEM_PROMPT = """\
You are an expert in text classification. Your task is to analyze user comments about Universal Basic Income (UBI) and categorize them into a more detailed stance classification system.

## Classification Categories
### 1. Pro-UBI (Support UBI)
- **{strong_support}:** The comment expresses strong endorsement of UBI without major concerns.
- **{conditional_support}:** The comment supports UBI only under specific conditions (e.g., alternative funding sources, specific implementation methods).
- **{rational_support}:** The comment supports UBI but raises practical considerations (e.g., economic feasibility, policy details).

### 2. Anti-UBI (Oppose UBI)
- **{strong_opposition}:** The comment strongly opposes UBI, often with negative language.
- **{conditional_opposition}:** The comment opposes UBI if certain conditions are met (e.g., if it requires high taxes).
- **{rational_opposition}:** The comment criticizes UBI but in a reasoned, non-hostile way.

### 3. Neutral / Uncertain Stance
- **{observational}:** The comment does not take a clear stance but expresses interest in further evaluation or experimentation.
- **{questioning}:** The comment does not state a stance but asks relevant questions or brings up discussion points.
- **{alternative_proposal}:** The comment does not support UBI but suggests a different approach (e.g., targeted subsidies instead of universal payments).

## Classification Instructions
- Read the comment carefully.
- Classify it into **one** of the categories above.
- If the comment fits multiple categories, choose the **best** match based on the overall sentiment.
- Do **not** assume a stance if it is unclear—use the "Neutral / Uncertain" categories when necessary.
- Provide a **one-sentence explanation** for your classification.

## Format Instructions
{format_instructions}\
"""

_USER_PROMPT = """\
User Comment:
####
{comment_text}
####\
"""

_CATEGORY_MINOR_TO_MAJOR = {
    STANCE_MINOR_CATEGORY.STRONG_SUPPORT: STANCE_MAJOR_CATEGORY.PRO_UBI.value,
    STANCE_MINOR_CATEGORY.CONDITIONAL_SUPPORT: STANCE_MAJOR_CATEGORY.PRO_UBI.value,
    STANCE_MINOR_CATEGORY.RATIONAL_SUPPORT: STANCE_MAJOR_CATEGORY.PRO_UBI.value,
    STANCE_MINOR_CATEGORY.STRONG_OPPONENT: STANCE_MAJOR_CATEGORY.ANTI_UBI.value,
    STANCE_MINOR_CATEGORY.CONDITIONAL_OPPONENT: STANCE_MAJOR_CATEGORY.ANTI_UBI.value,
    STANCE_MINOR_CATEGORY.RATIONAL_OPPONENT: STANCE_MAJOR_CATEGORY.ANTI_UBI.value,
    STANCE_MINOR_CATEGORY.OBSERVATIONAL: STANCE_MAJOR_CATEGORY.NEUTRAL.value,
    STANCE_MINOR_CATEGORY.QUESTIONING: STANCE_MAJOR_CATEGORY.NEUTRAL.value,
    STANCE_MINOR_CATEGORY.ALTERNATIVE_PROPOSAL: STANCE_MAJOR_CATEGORY.NEUTRAL.value,
}


class _ModelResponse(BaseModel):
    reasoning: str = Field(description="The reasoning behind the classification")
    category: STANCE_MINOR_CATEGORY = Field(description="The selected category")


class StanceClassifier(object):
    def __init__(
        self, model_name: str = "gpt-4o-mini-2024-07-18", temperature: float = 0.0
    ):
        self._llm = create_openai_llm(model_name=model_name, temperature=temperature)
        self._parser = JsonOutputParser(pydantic_object=_ModelResponse)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _USER_PROMPT),
            ]
        ).partial(
            strong_support=STANCE_MINOR_CATEGORY.STRONG_SUPPORT,
            conditional_support=STANCE_MINOR_CATEGORY.CONDITIONAL_SUPPORT,
            rational_support=STANCE_MINOR_CATEGORY.RATIONAL_SUPPORT,
            strong_opposition=STANCE_MINOR_CATEGORY.STRONG_OPPONENT,
            conditional_opposition=STANCE_MINOR_CATEGORY.CONDITIONAL_OPPONENT,
            rational_opposition=STANCE_MINOR_CATEGORY.RATIONAL_OPPONENT,
            observational=STANCE_MINOR_CATEGORY.OBSERVATIONAL,
            questioning=STANCE_MINOR_CATEGORY.QUESTIONING,
            alternative_proposal=STANCE_MINOR_CATEGORY.ALTERNATIVE_PROPOSAL,
            format_instructions=self._parser.get_format_instructions(),
        )
        self._chain = self._prompt | self._llm | self._parser

    def _classify(self, comment_text: str) -> StanceClassificationResult:
        chain_output = self._chain.invoke(
            {
                "comment_text": comment_text,
            }
        )

        minor_category = chain_output.get("category")
        result = StanceClassificationResult(
            major_category=_CATEGORY_MINOR_TO_MAJOR[minor_category],
            minor_category=minor_category,
            detail={"reasoning": chain_output.get("reasoning")},
        )

        return result
