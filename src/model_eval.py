import pandas as pd
import torch
import json
import re
import asyncio
import logging
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .evaluate.judge import evaluate_domains
from .prompt import DOMAIN_GENERATION_PROMPT


def generate_domains(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, description: str) -> List[str]:
    prompt: str = DOMAIN_GENERATION_PROMPT.format(description=description)
    test_input: str = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    input_token_length: int = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_token_ids = outputs[0, input_token_length:]
    response: str = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    domains: List[str] = []
    for line in response.split('\n'):
        line = line.strip()
        if line:
            domain: str = re.sub(r'^\d+\.\s*', '', line)
            domain = re.sub(r'[^a-zA-Z0-9]', '', domain.lower())
            if domain and len(domain) > 2:
                domains.append(domain)
    
    return domains


async def evaluate_model(version: str) -> None:
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    model_path: str = f"models/model_{version}"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    except Exception as e:
        logging.error(f"Failed to load model {version}: {e}")
        raise
    
    try:
        test_df: pd.DataFrame = pd.read_csv('data/test_set.csv')
    except FileNotFoundError:
        logging.error("Test set not found at data/test_set.csv")
        raise
    
    rows_with_domains: List[Dict[str, Any]] = []
    for idx, row in test_df.iterrows():
        description: str = row['description']
        print(f"Generating domains {idx+1}/{len(test_df)}: {description[:50]}...")
        
        domains: List[str] = generate_domains(model, tokenizer, description)
        rows_with_domains.append({
            'idx': idx,
            'description': description,
            'domains': domains
        })
    
    results: List[Dict[str, Any]] = []
    for row_data in rows_with_domains:
        idx: int = row_data['idx']
        description: str = row_data['description']
        domains: List[str] = row_data['domains']
        
        print(f"Evaluating domains {idx+1}/{len(test_df)}: {description[:50]}...")
        
        evaluation = await evaluate_domains(description, domains)

        category_counts: Dict[str, int] = {
            "good": 0,
            "ok": 0,
            "random_word": 0,
            "too_long": 0,
            "other_failure": 0,
            "inappropriate": 0
        }

        for eval_item in evaluation.evaluations:
            category: str = eval_item.scores.domain_category
            category_counts[category] += 1

        results.append({
            'description': description,
            'domains': json.dumps(domains),
            'overall_category': evaluation.description_category,
            'is_appropriate': evaluation.is_appropriate,
            'good_count': category_counts['good'],
            'ok_count': category_counts['ok'],
            'random_word_count': category_counts['random_word'],
            'too_long_count': category_counts['too_long'],
            'other_failure_count': category_counts['other_failure'],
            'inappropriate': category_counts['inappropriate'],
            'average_score': evaluation.average_score
        })
    
    results_df: pd.DataFrame = pd.DataFrame(results)
    output_path: str = f'data/model_{version}-results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def run_evaluation(version: str) -> None:
    asyncio.run(evaluate_model(version))