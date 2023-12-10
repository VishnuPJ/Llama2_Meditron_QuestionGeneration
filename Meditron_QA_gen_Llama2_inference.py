# !pip install -q  torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate


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
from peft import PeftModel, PeftConfig

from huggingface_hub import login
login(token="hf_token")

context = """Arthritis is the swelling and tenderness of one or more joints. The main symptoms of arthritis are joint pain and stiffness, which typically worsen with age. The most common types of arthritis are osteoarthritis and rheumatoid arthritis.
Osteoarthritis causes cartilage — the hard, slippery tissue that covers the ends of bones where they form a joint — to break down. Rheumatoid arthritis is a disease in which the immune system attacks the joints, beginning with the lining of joints.
Uric acid crystals, which form when there's too much uric acid in your blood, can cause gout. Infections or underlying disease, such as psoriasis or lupus, can cause other types of arthritis.
Treatments vary depending on the type of arthritis. The main goals of arthritis treatments are to reduce symptoms and improve quality of life."""


sys_prompt = "You are a subject matter expert(SME) in medical domain.\
    Given a context ,create multiple-choice questions (MCQs) based on the symptoms,\
    causes, treatments, or characteristics associated within the context. Formulate\
    four options, one of which should be the most likely or accurate choice based on the provided information.\
    Also specify which is right choice."


prompt_2 = f"""<s> [INST] <<SYS>> {sys_prompt} <</SYS>> { context } [/INST] """


# Model and tokenizer names
base_model_name = "epfl-llm/meditron-7b"
refined_model = "meditron-7b-MCQ" #You can give it your own name
# refined_model = r"checkpoint_750_loss_0.6841")

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
    bnb_4bit_use_double_quant=False,
    
)


# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    # device_map="auto"
    # device_map={"": "cpu"},
    device_map={"": 0}
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

model = PeftModel.from_pretrained(base_model,
    refined_model
    )
# model = model.merge_and_unload()

#model.save_pretrained(r"Merged_Base+Adapter", safe_serialization=True)
#llama_tokenizer.save_pretrained(r"Merged_Base+Adapter")


model = model.to("cuda")
model.eval()

inputs = llama_tokenizer(prompt_2, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=100
    )
    print(
        llama_tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0]
    )
