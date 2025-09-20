from pydantic import BaseModel, ConfigDict


class DomainScores(BaseModel):
    model_config = ConfigDict(extra='forbid') #`"additionalProperties": false` for OpenAI's strict mode

    relevance: int
    creativity: int
    brandability: int
    conciseness: int


class DomainEvaluation(BaseModel):
    model_config = ConfigDict(extra='forbid') #`"additionalProperties": false` for OpenAI's strict mode

    domain: str
    scores: DomainScores

    @property
    def total_score(self) -> float:
        total = self.scores.relevance + self.scores.creativity + self.scores.brandability + self.scores.conciseness
        return (total / 4) / 5.0  # Convert from 1-5 scale to 0-1 scale


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra='forbid') #`"additionalProperties": false` for OpenAI's strict mode

    business_description: str
    evaluations: list[DomainEvaluation]
    
    @property
    def average_score(self) -> float:
        if not self.evaluations:
            return 0.0
        
        total_score = sum(eval.total_score for eval in self.evaluations)
        return total_score / len(self.evaluations)