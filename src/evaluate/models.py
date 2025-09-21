from pydantic import BaseModel, ConfigDict, Field, computed_field
from typing import Literal


class DomainScores(BaseModel):
    model_config = ConfigDict(extra='forbid') #`"additionalProperties": false` for OpenAI's strict mode

    relevance: int
    creativity: int
    brandability: int
    conciseness: int
    domain_category: Literal["good", "ok", "random_word", "too_long", "other_failure", "inappropriate"]


class DomainEvaluation(BaseModel):
    model_config = ConfigDict(extra='forbid') #`"additionalProperties": false` for OpenAI's strict mode

    domain: str
    scores: DomainScores

    @property
    def total_score(self) -> float:
        total = self.scores.relevance + self.scores.creativity + self.scores.brandability + self.scores.conciseness
        return total / (4 * 5.0)  # Convert from 0-5 scale to 0-1 scale


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra='forbid') #`"additionalProperties": false` for OpenAI's strict mode

    business_description: str
    evaluations: list[DomainEvaluation]
    is_appropriate: bool

    # This won't be included in the schema but can be set manually
    description_category: str

    @property
    def average_score(self) -> float:
        if not self.evaluations:
            return 0.0
        
        total_score = sum(eval.total_score for eval in self.evaluations)
        return total_score / len(self.evaluations)