
# pip install -q  torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate

import os
os.environ['TRANSFORMERS_CACHE'] = r"HF_cache_path"


import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer

from huggingface_hub import login
login(token="hf_token")



sys_prompt = "You are a subject matter expert(SME) in medical domain.\
    Given a context ,create multiple-choice questions (MCQs) based on the symptoms,\
    causes, treatments, or characteristics associated within the context. Formulate\
    four options, one of which should be the most likely or accurate choice based on the provided information.\
    Also specify which is right choice."
    
def generate_prompt(x):
    output_texts = []

    # for idx, x in dFrame.iterrows():
    for i in range(len(x['question'])):
      question = '\n{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}\nRight_Ans:{}\n'.format(x['question'][i], x['opa'][i], x['opb'][i], x['opc'][i], x['opd'][i], x['answer'][i])
      prompt = f"""<s> [INST] <<SYS>> {sys_prompt} <</SYS>> { x['new_context'][i] } [/INST] { question }  </s>"""
      output_texts.append(prompt)
    return output_texts


# Dataset
data_name = "medmcqa"
training_data = load_dataset(data_name, split="train")
# dataset = load_dataset('csv', data_files={'train': ['train_data.csv'], 'val': 'val_data.csv'})



# Model and tokenizer names
base_model_name = "epfl-llm/meditron-7b"
refined_model = "meditron-7b-MCQ" #You can give it your own name

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, 
                                                trust_remote_code=True,
                                                use_auth_token=True)

llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)


# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    # device_map="auto"
    device_map={"": 0}
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1


# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    # save_total_limit=10,
    # load_best_model_at_end=True,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    # train_dataset=dataset['train'],
    train_dataset=training_data
    peft_config=peft_parameters,
    max_seq_length=1024,
    tokenizer=llama_tokenizer,
    formatting_func=generate_prompt,
    args=train_params,
    # eval_dataset=dataset['val']
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)
fine_tuning.tokenizer.save_pretrained(refined_model)

#Evaluation
# from tensorboard import notebook
# log_dir = "results/runs"
# notebook.start("--logdir {} --port 4000".format(log_dir))