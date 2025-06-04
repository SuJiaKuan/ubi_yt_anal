from enum import Enum


class STANCE_MAJOR_CATEGORY(str, Enum):
    PRO_UBI = "Pro-UBI"
    ANTI_UBI = "Anti-UBI"
    NEUTRAL = "Neutral"


class STANCE_MINOR_CATEGORY(str, Enum):
    STRONG_SUPPORT = "Strong Support"
    CONDITIONAL_SUPPORT = "Conditional Support"
    RATIONAL_SUPPORT = "Rational Support"
    STRONG_OPPONENT = "Strong Opposition"
    CONDITIONAL_OPPONENT = "Conditional Opposition"
    RATIONAL_OPPONENT = "Rational Opposition"
    OBSERVATIONAL = "Observational / Open-Minded"
    QUESTIONING = "Questioning / Discussion-Oriented"
    ALTERNATIVE_PROPOSAL = "Alternative Proposal"


class TOPIC_MAJOR_TAG(str, Enum):
    POLITICS = "Politics"
    ECONOMICS = "Economics"
    SOCIETY = "Society"
    PHILOSOPHY_AND_ETHICS = "Philosophy & Ethics"


class TOPIC_MINOR_TAG(str, Enum):
    # Politics
    ROLE_OF_GOVERNMENT = "Role of Government"
    COMPARISON_WITH_SOCIAL_POLICIES = "Comparison with Social Policies"
    IDEOLOGIES = "Ideologies"
    FEASIBILITY_AND_GOVERNANCE = "Feasibility & Governance"
    # Economics
    INFLATION_AND_COST_OF_LIVING = "Inflation & Cost of Living"
    TAXATION_AND_BUDGETING = "Taxation & Budgeting"
    LABOR_MARKET_AND_EMPLOYMENT = "Labor Market & Employment"
    ECONOMIC_GROWTH_AND_PRODUCTIVITY = "Economic Growth & Productivity"
    # Society
    POVERTY_AND_WEALTH_DISTRIBUTION = "Poverty & Wealth Distribution"
    WORK_ETHIC_AND_MOTIVATION = "Work Ethic & Motivation"
    SOCIAL_STABILITY_AND_CRIME_RATE = "Social Stability & Crime Rate"
    MENTAL_HEALTH_AND_WELL_BEING = "Mental Health & Well-being"
    # Philosophy & Ethics
    EQUALITY_AND_FAIRNESS = "Equality & Fairness"
    TECHNOLOGY_AND_THE_FUTURE = "Technology & The Future"
    HUMAN_NATURE_AND_BEHAVIOR = "Human Nature & Behavior"
    FREEDOM_VS_DEPENDENCY = "Freedom vs. Dependency"


class ARGUMENT_STANCE_CATEGORY(str, Enum):
    PRO = "Pro"
    CON = "Con"
    NONE = "None"


class ARGUMENT_MAJOR_CATEGORY(str, Enum):
    SOCIETY_ETHICS = "Society & Ethics"
    POLITICS_POLICY = "Politics & Policy"
    ECONOMY_INSTITUTIONS = "Economy & Institutions"
    PERSONAL_CHOICE = "Personal Choice"


class ARGUMENT_LABEL(str, Enum):
    # Society & Ethics
    HUMAN_NATURE_AND_LAZINESS = "Human Nature & Laziness"
    FREEDOM = "Freedom"
    FAIRNESS = "Fairness"
    SOCIAL_SAFETY_NET = "Social Safety Net"
    PRECARIAT = "Precariat"
    WELLBEING = "Wellbeing"
    POVERTY_TRAP = "Poverty Trap"
    STIGMA = "Stigma"
    EDUCATION = "Education"
    SURVIVAL = "Survival"
    RESPONSIBILITY = "Responsibility"
    FREERIDING = "Freeriding"
    # Politics & policy
    BUREAUCRACY = "Bureaucracy"
    POLITICAL_FEASIBILITY = "Political Feasibility"
    SOCIAL_WELFARE = "Social Welfare"
    IMMIGRATION = "Immigration"
    CAPITALISM = "Capitalism"
    COMMUNISM_AND_SOCIALISM = "Communism and Socialism"
    UNCONDITIONALITY = "Unconditionality"
    UNIVERSALITY = "Universality"
    # Economy & Institutions
    REDISTRIBUTION = "Redistribution"
    INFLATION_AND_COST_OF_LIVING = "Inflation and Cost of Living"
    AI_AND_AUTOMATION = "AI and Automation"
    ECONOMIC_FEASIBILITY_AND_SUSTAINABILITY = "Economic Feasibility and Sustainability"
    FUNDING_AND_BUDGET = "Funding and Budget"
    DEREGULATION = "Deregulation"
    ACTIVATION = "Activation"
    LABOR_MARKET_IMPACT = "Labor Market Impact"
    SOCIAL_INNOVATION = "Social Innovation"
    # Personal Choice
    EFFORT = "Effort"
    SOCIAL_PARTICIPATION = "Social Participation"
    LIFE_MARGIN = "Life Margin"
    LIFE_MEANING = "Life Meaning"
    FAMILY_AND_PARENTING = "Family and Parenting"
    SELF_REALIZATION = "Self-Realization"
    GUILT = "Guilt"
    LEISURE_AND_LIFESTYLE = "Leisure and Lifestyle"
    ENTREPRENEURSHIP = "Entrepreneurship"
    SECURITY = "Security"
    LIBERATION = "Liberation"
