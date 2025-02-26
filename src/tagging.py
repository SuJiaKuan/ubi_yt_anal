from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pydantic import Field

from src.const import TOPIC_MINOR_TAG
from src.shape import TaggingResult
from src.model import create_openai_llm

_TOPIC_SYSTEM_PROMPT = """\
You are an expert in text tagging. Your task is to analyze user comments (from YouTube videos) about Universal Basic Income (UBI) and assign appropriate topic tags.

Each comment may have **0, 1, or multiple tags**, based on its content. 

## Topic Tags and Descriptions:

### **Politics**
- **{role_of_government}:** Discusses the role of the state in providing basic income, whether it should be a large welfare state or a minimal government.
- **{comparison_with_social_policies}:** Compares UBI with other policies like healthcare, education, and pensions.
- **{ideologies}:** Mentions capitalism, socialism, communism, or other political ideologies in relation to UBI.
- **{feasibility_and_governance}:** Discusses whether governments can implement UBI effectively, including concerns about bureaucracy and policy execution.


### **Economics**
- **{inflation_and_cost_of_living}:** Discusses whether UBI will increase inflation or impact purchasing power.
- **{taxation_and_budgeting}:** Talks about funding UBI through taxes or redistribution of wealth.
- **{labor_market_and_employment}:** Discusses how UBI will affect jobs, salaries, or workforce participation.
- **{economic_growth_and_productivity}:** Mentions UBI’s impact on economic efficiency, GDP, or productivity.

### **Society**
- **{poverty_and_wealth_distribution}:** Discusses UBI’s potential to reduce poverty or wealth inequality.
- **{work_ethic_and_motivation}:** Raises concerns about whether UBI will discourage people from working.
- **{social_stability_and_crime_rate}:** Talks about UBI’s effects on crime rates, social cohesion, or societal stability.
- **{mental_health_and_well_being}:** Discusses how UBI could impact mental health, happiness, or stress levels.

### **Philosophy & Ethics**
- **{equality_and_fairness}:** Argues whether UBI is just or unjust from a moral standpoint.
- **{technology_and_the_future}:** Discusses how automation and AI may make UBI necessary or inevitable.
- **{human_nature_and_behavior}:** Questions how humans will adapt to a world with guaranteed income.
- **{freedom_vs_dependency}:** Discusses whether UBI makes people more independent or overly reliant on government support.

## Tagging Instructions
- Read the comment carefully.
- Provide a **detailed reasoning** for your tagging.
- Assign **0, 1, or multiple tags** from the list above.
- If the comment is irrelevant or does not fit any topic, provide an empty tag list.

## Format Instructions
{format_instructions}\
"""

_USER_PROMPT = """\
User Comment:
####
{comment_text}
####\
"""


class _TopicModelResponse(BaseModel):
    reasoning: str = Field(description="The reasoning behind the tagging")
    tags: list[TOPIC_MINOR_TAG] = Field(description="The selected tags")


class TopicTagger(object):
    def __init__(
        self, model_name: str = "gpt-4o-mini-2024-07-18", temperature: float = 0.0
    ):
        self._llm = create_openai_llm(model_name=model_name, temperature=temperature)
        self._parser = JsonOutputParser(pydantic_object=_TopicModelResponse)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _TOPIC_SYSTEM_PROMPT),
                ("human", _USER_PROMPT),
            ]
        ).partial(
            format_instructions=self._parser.get_format_instructions(),
            role_of_government=TOPIC_MINOR_TAG.ROLE_OF_GOVERNMENT,
            comparison_with_social_policies=TOPIC_MINOR_TAG.COMPARISON_WITH_SOCIAL_POLICIES,
            ideologies=TOPIC_MINOR_TAG.IDEOLOGIES,
            feasibility_and_governance=TOPIC_MINOR_TAG.FEASIBILITY_AND_GOVERNANCE,
            inflation_and_cost_of_living=TOPIC_MINOR_TAG.INFLATION_AND_COST_OF_LIVING,
            taxation_and_budgeting=TOPIC_MINOR_TAG.TAXATION_AND_BUDGETING,
            labor_market_and_employment=TOPIC_MINOR_TAG.LABOR_MARKET_AND_EMPLOYMENT,
            economic_growth_and_productivity=TOPIC_MINOR_TAG.ECONOMIC_GROWTH_AND_PRODUCTIVITY,
            poverty_and_wealth_distribution=TOPIC_MINOR_TAG.POVERTY_AND_WEALTH_DISTRIBUTION,
            work_ethic_and_motivation=TOPIC_MINOR_TAG.WORK_ETHIC_AND_MOTIVATION,
            social_stability_and_crime_rate=TOPIC_MINOR_TAG.SOCIAL_STABILITY_AND_CRIME_RATE,
            mental_health_and_well_being=TOPIC_MINOR_TAG.MENTAL_HEALTH_AND_WELL_BEING,
            equality_and_fairness=TOPIC_MINOR_TAG.EQUALITY_AND_FAIRNESS,
            technology_and_the_future=TOPIC_MINOR_TAG.TECHNOLOGY_AND_THE_FUTURE,
            human_nature_and_behavior=TOPIC_MINOR_TAG.HUMAN_NATURE_AND_BEHAVIOR,
            freedom_vs_dependency=TOPIC_MINOR_TAG.FREEDOM_VS_DEPENDENCY,
        )
        self._chain = self._prompt | self._llm | self._parser

    def tag(self, comment_text: str) -> dict:
        chain_output = self._chain.invoke(
            {
                "comment_text": comment_text,
            }
        )

        result = TaggingResult(
            tags=chain_output.get("tags"),
            detail={"reasoning": chain_output.get("reasoning")},
        )

        return result
