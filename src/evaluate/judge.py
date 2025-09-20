import json
import logging
from openai import AsyncOpenAI
from ..settings import config
from .models import DomainEvaluation, DomainScores, EvaluationResult

logger = logging.getLogger(__name__)


async def evaluate_domains(business_description: str, domains: list[str]) -> EvaluationResult:
    client = AsyncOpenAI(api_key=config.openai_credentials)
    
    prompt = create_evaluation_prompt(business_description, domains)
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert domain name evaluator. You will score domain names based on specific criteria."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent scoring
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "domain_evaluation",
                    "schema": EvaluationResult.model_json_schema(),
                    "strict": True # Slightly slower - but ensures correctness
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        evaluation_result = EvaluationResult(**result)
        return evaluation_result
        
    except Exception as e:
        logger.exception(f"Error evaluating domains: {e}")
        
        # Fallback with 0 scores for all domains
        fallback_evaluations = []
        for domain in domains:
            fallback_scores = DomainScores(
                relevance=0,
                creativity=0,
                brandability=0,
                conciseness=0
            )
            fallback_evaluations.append(DomainEvaluation(
                domain=domain,
                scores=fallback_scores
            ))
        
        return EvaluationResult(
            business_description=business_description,
            evaluations=fallback_evaluations
        )


def create_evaluation_prompt(business_description: str, domains: list[str]) -> str:
    domains_text = "\n".join([f"- {domain}" for domain in domains])

    prompt = f"""
Please evaluate the following domain names for this business:

BUSINESS DESCRIPTION: {business_description}

DOMAIN NAMES TO EVALUATE:
{domains_text}

SCORING CRITERIA (Rate each domain on a scale of 1-5 for each criterion):

1. **Relevance (1-5)**: How well does the domain name match the business? Does it clearly relate to the industry, service, or purpose?
   - 5: Perfect match, immediately obvious connection
   - 4: Strong connection, very relevant
   - 3: Moderate connection, somewhat relevant
   - 2: Weak connection, marginally relevant
   - 1: No clear connection, irrelevant

2. **Creativity (1-5)**: Is the domain name memorable and unique? Does it stand out from competitors?
   - 5: Highly creative, very memorable, unique approach
   - 4: Creative and memorable
   - 3: Moderately creative, somewhat memorable
   - 2: Low creativity, generic feeling
   - 1: Not creative, very generic

3. **Brandability (1-5)**: Does it sound like a real brand? Is it professional and trustworthy?
   - 5: Excellent brand potential, very professional
   - 4: Good brand potential, sounds professional
   - 3: Moderate brand potential
   - 2: Limited brand potential, questionable professionalism
   - 1: Poor brand potential, unprofessional

4. **Conciseness (1-5)**: Is it short and easy to type? Is it memorable and not overly complex?
   - 5: Very short and simple (≤8 characters)
   - 4: Short and simple (9-12 characters)
   - 3: Moderate length (13-16 characters)
   - 2: Somewhat long (17-20 characters)
   - 1: Too long or complex (>20 characters)

Evaluate each domain thoroughly and return the results in the required JSON format.
"""

    return prompt
