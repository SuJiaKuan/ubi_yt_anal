from dataclasses import dataclass


@dataclass
class StanceClassificationResult:
    major_category: str
    minor_category: str
    detail: dict


@dataclass
class TaggingResult:
    tags: list[str]
    detail: dict


@dataclass
class ScoringResult:
    score: int
    detail: dict


@dataclass
class FramedArgumentResult:
    reason: str
    label: str
    stance: str
