import json, argparse, torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DATA = "dpo_collapse.jsonl"   # ou "dpo_control.jsonl"
OUT  = "out_collapse"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])

ds = load_dataset("json", data_files=DATA, split="train")

#ds = ds.select(range(32))

cfg = DPOConfig(
    output_dir=OUT, beta=0.1, learning_rate=5e-5,
    num_train_epochs=3, per_device_train_batch_size=1,
    gradient_accumulation_steps=8, gradient_checkpointing=True,
    max_length=512, bf16=True, logging_steps=10,
    save_strategy="epoch", report_to="none")


trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                     processing_class=tok, peft_config=lora)
trainer.train()
trainer.save_model(OUT)  # salva só o adapter LoRA