import pandas as pd
import json
import torch
import logging
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import Dataset
from .prompt import DOMAIN_GENERATION_PROMPT


def train_model(version: str) -> None:
    dataset: pd.DataFrame = pd.read_csv(f'data/dataset_{version}.csv')
    
    for idx, row in dataset.iterrows():
        try:
            suggestions: List[str] = json.loads(row['suggestions'])
            if not isinstance(suggestions, list) or (len(suggestions) != 5 and len(suggestions) != 0):
                logging.error(f"Row {idx}: suggestions validation failed")
                raise ValueError(f"Dataset {version} validation failed")
        except json.JSONDecodeError:
            logging.error(f"Row {idx}: invalid JSON in suggestions")
            raise ValueError(f"Dataset {version} validation failed")
    
    def format_dataset(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        texts: List[str] = []
        for description, suggestions_json_str in zip(examples['description'], examples['suggestions']):
            try:
                suggestions_list: List[str] = json.loads(suggestions_json_str)
            except json.JSONDecodeError:
                suggestions_list = []
            
            suggestions_formatted: str = "\n".join(suggestions_list)
            prompt: str = DOMAIN_GENERATION_PROMPT.format(description=description)
            text: str = f"""<s>[INST] {prompt} [/INST] {suggestions_formatted}</s>"""
            texts.append(text)
        return {"text": texts}
    
    full_dataset = Dataset.from_pandas(dataset)
    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train'].map(format_dataset, batched=True, remove_columns=split['train'].column_names)
    val_dataset = split['test'].map(format_dataset, batched=True, remove_columns=split['test'].column_names)
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    training_config = SFTConfig(
        output_dir=f"models/model_{version}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=10,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to=None,
        remove_unused_columns=False,
        max_length=1024,
        fp16=False if torch.backends.mps.is_available() else True,
        bf16=True if torch.backends.mps.is_available() else False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    trainer.train()
    trainer.save_model(f"models/model_{version}")
    tokenizer.save_pretrained(f"models/model_{version}")


def test_model(version: str) -> None:
    try:
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            f"models/model_{version}",
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(f"models/model_{version}")
        
        description: str = "Artisanal coffee roastery with single-origin beans"
        prompt: str = DOMAIN_GENERATION_PROMPT.format(description=description)
        test_input: str = f"<s>[INST] {prompt} [/INST]"
        
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response: str = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Test response for {version}: {response}")
        
    except Exception as e:
        logging.error(f"Test failed for {version}: {e}")
        raise