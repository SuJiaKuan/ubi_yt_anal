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
