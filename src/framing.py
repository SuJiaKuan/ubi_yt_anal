from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pydantic import Field

from typing import List

from src.const import ARGUMENT_STANCE_CATEGORY
from src.const import ARGUMENT_LABEL
from src.shape import FramedArgumentResult
from src.model import create_openai_llm


_ARGUMENT_LABEL_DESCRIPTION = {
    # Society & Ethics
    ARGUMENT_LABEL.HUMAN_NATURE_AND_LAZINESS: "Discusses whether UBI promotes or counters human tendencies such as laziness, motivation, or personal responsibility.",
    ARGUMENT_LABEL.FREEDOM: "Focuses on whether UBI increases or undermines personal freedom and life choices.",
    ARGUMENT_LABEL.FAIRNESS: "Concerns whether UBI is perceived as fair or unfair in terms of distribution, effort, or opportunity.",
    ARGUMENT_LABEL.SOCIAL_SAFETY_NET: "Addresses whether UBI serves as a reliable or problematic form of basic social protection.",
    ARGUMENT_LABEL.PRECARIAT: "Relates to UBI's effects on those in unstable or precarious forms of work.",
    ARGUMENT_LABEL.WELLBEING: "Frames UBI in terms of psychological, emotional, or physical wellbeing.",
    ARGUMENT_LABEL.POVERTY_TRAP: "Discusses whether UBI helps individuals escape cycles of poverty or dependency.",
    ARGUMENT_LABEL.STIGMA: "Refers to whether UBI reduces or reinforces stigma against welfare recipients or non-working individuals.",
    ARGUMENT_LABEL.EDUCATION: "Evaluates whether UBI enables access to education or affects educational motivation.",
    ARGUMENT_LABEL.SURVIVAL: "Connects UBI with basic existential needs and minimal living conditions.",
    ARGUMENT_LABEL.RESPONSIBILITY: "Explores whether UBI supports or discourages a sense of duty, work ethic, or accountability.",
    ARGUMENT_LABEL.FREERIDING: "Frames UBI as encouraging or preventing exploitation of social resources without contribution.",
    # Politics & Policy
    ARGUMENT_LABEL.BUREAUCRACY: "Looks at whether UBI reduces administrative complexity or contributes to it.",
    ARGUMENT_LABEL.POLITICAL_FEASIBILITY: "Examines whether UBI is realistically implementable given current political systems.",
    ARGUMENT_LABEL.SOCIAL_WELFARE: "Relates UBI to traditional welfare systems, either as complement or replacement.",
    ARGUMENT_LABEL.IMMIGRATION: "Considers whether UBI might attract or repel migration flows, or burden native systems.",
    ARGUMENT_LABEL.CAPITALISM: "Places UBI in the context of capitalist critiques or support structures.",
    ARGUMENT_LABEL.COMMUNISM_AND_SOCIALISM: "Frames UBI as a socialist/communist policy or as a departure from such ideologies.",
    ARGUMENT_LABEL.UNCONDITIONALITY: "Evaluates whether the unconditional nature of UBI—being given without means testing, work requirements, or behavioral conditions—is seen as a positive feature that empowers autonomy, or a negative one that reduces accountability or efficiency.",
    ARGUMENT_LABEL.UNIVERSALITY: "Focuses on the implications of providing UBI to all versus targeted recipients.",
    # Economy & Institutions
    ARGUMENT_LABEL.REDISTRIBUTION: "Discusses UBI as a means of redistributing wealth or burdening productive contributors.",
    ARGUMENT_LABEL.INFLATION_AND_COST_OF_LIVING: "Concerns whether UBI may cause or cushion inflation or higher costs of living.",
    ARGUMENT_LABEL.AI_AND_AUTOMATION: "Relates UBI to job losses or transitions caused by technological change.",
    ARGUMENT_LABEL.ECONOMIC_FEASIBILITY_AND_SUSTAINABILITY: "Evaluates the long-term viability and affordability of UBI.",
    ARGUMENT_LABEL.FUNDING_AND_BUDGET: "Questions where the money for UBI comes from and how budgets would adjust.",
    ARGUMENT_LABEL.DEREGULATION: "Frames UBI as enabling a smaller, less intrusive government or social system.",
    ARGUMENT_LABEL.ACTIVATION: "Discusses whether UBI encourages work participation or leads to withdrawal from labor.",
    ARGUMENT_LABEL.LABOR_MARKET_IMPACT: "Assesses how UBI might affect wages, hiring, job creation, or work incentives.",
    ARGUMENT_LABEL.SOCIAL_INNOVATION: "Discusses whether UBI enables or promotes social innovation, such as new forms of work, community models, or welfare reform",
    # Personal Choice
    ARGUMENT_LABEL.EFFORT: "Focuses on whether UBI affects people's willingness to try hard or contribute.",
    ARGUMENT_LABEL.SOCIAL_PARTICIPATION: "Relates to how UBI might enable or weaken civic engagement or social involvement.",
    ARGUMENT_LABEL.LIFE_MARGIN: "Explores whether UBI increases time and psychological space for non-work pursuits.",
    ARGUMENT_LABEL.LIFE_MEANING: "Connects UBI to individuals' sense of purpose, identity, or fulfillment.",
    ARGUMENT_LABEL.FAMILY_AND_PARENTING: "Discusses UBI's potential influence on caregiving, parenting, or family dynamics.",
    ARGUMENT_LABEL.SELF_REALIZATION: "Frames UBI as enabling self-actualization, personal growth, or creative goals.",
    ARGUMENT_LABEL.GUILT: "Concerns whether recipients feel undeserving, ashamed, or morally conflicted about UBI.",
    ARGUMENT_LABEL.LEISURE_AND_LIFESTYLE: "Examines UBI in relation to lifestyle choices and time use beyond employment.",
    ARGUMENT_LABEL.ENTREPRENEURSHIP: "Looks at whether UBI supports risk-taking, innovation, or small business creation.",
    ARGUMENT_LABEL.SECURITY: "Frames UBI as promoting financial, social, or emotional stability.",
    ARGUMENT_LABEL.LIBERATION: "Connects UBI with freedom from oppressive systems or constraints (e.g., job dependency).",
}


_SYSTEM_PROMPT = """\
You are an expert in argument-based discourse analysis.

For each of the following argument labels, determine whether the given comment expresses a:
- **{stance_pro}** stance: the concept is positively framed in relation to Universal Basic Income (UBI)
- **{stance_con}** stance: the concept is negatively framed in relation to UBI
- **{stance_none}**: the concept is not mentioned or not clearly linked to UBI in this comment

## Argument Labels (with Explanations):
{argument_labels}

## Format Instructions
{format_instructions}\
"""

_USER_PROMPT = """\
User Comment:
####
{comment_text}
####\
"""


class _FramedArgument(BaseModel):
    label: str = Field(description="The argument label being evaluated")
    reason: str = Field(
        description=f"A short explanation (1-2 sentences) describing why the stance was assigned. "
        f"If the stance is **{ARGUMENT_STANCE_CATEGORY.NONE.value}**, explain why the argument label is not relevant or mentioned. "
        f"This field ensures interpretability and transparency in labeling."
    )
    stance: str = Field(description="The stance of the argument label")


class _ModelResponse(BaseModel):
    results: List[_FramedArgument]


class ArgumentFramer(object):
    def __init__(
        self, model_name: str = "gpt-4.1-mini-2025-04-14", temperature: float = 0.0
    ):
        self._llm = create_openai_llm(model_name=model_name, temperature=temperature)
        self._parser = JsonOutputParser(pydantic_object=_ModelResponse)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _USER_PROMPT),
            ]
        ).partial(
            format_instructions=self._parser.get_format_instructions(),
            stance_pro=ARGUMENT_STANCE_CATEGORY.PRO.value,
            stance_con=ARGUMENT_STANCE_CATEGORY.CON.value,
            stance_none=ARGUMENT_STANCE_CATEGORY.NONE.value,
            argument_labels="\n".join(
                [
                    f"- **{label.value}**: {desc}"
                    for label, desc in _ARGUMENT_LABEL_DESCRIPTION.items()
                ]
            ),
        )
        self._chain = self._prompt | self._llm | self._parser

    def frame(self, comment_text: str) -> dict:
        chain_output = self._chain.invoke(
            {
                "comment_text": comment_text,
            }
        )

        results = [
            FramedArgumentResult(
                label=r.get("label"),
                stance=r.get("stance"),
                reason=r.get("reason"),
            )
            for r in chain_output.get("results")
            if r.get("stance")
            in [
                ARGUMENT_STANCE_CATEGORY.PRO.value,
                ARGUMENT_STANCE_CATEGORY.CON.value,
            ]
        ]

        return results
