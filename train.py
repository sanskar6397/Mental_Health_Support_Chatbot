from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# Load dataset
dataset = load_dataset("json", data_files="data/finetune.jsonl", split="train")

# Format dataset into prompt → response style
def format_prompt(example):
    prompt = f"<s>[INST] <<SYS>>\n{example['instruction']}\n<</SYS>>\n\n{example['input']} [/INST]"
    return {"prompt": prompt, "response": example["output"]}

dataset = dataset.map(format_prompt)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="llama2-access")

def tokenize(example):
    full_text = example["prompt"] + " " + example["response"]
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize)

# Load model in 4bit mode
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto",
    token="llama2-access"
)

model = prepare_model_for_kbit_training(model)

# Apply LoRA (efficient fine-tuning)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training setup
args = TrainingArguments(
    output_dir="models/llama2-cbt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=1,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()

# Save model
model.save_pretrained("models/llama2-cbt")
tokenizer.save_pretrained("models/llama2-cbt")
