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
