from dataclasses import dataclass


@dataclass
class StanceClassificationResult:
    major_category: str
    minor_category: str
    detail: dict


@dataclass
class ScoringResult:
    score: int
    detail: dict
