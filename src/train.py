from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import pandas as pd
import wandb

def formatting_prompts_func(row):
    """ 各レコードからプロンプトを生成 """
    prompt_template = """
    ### Instruction:
    {} 

    ### Input:
    {} 

    ### Output:
    {}"""

    instruction = "Write a abstract for the following scientific paper."
    full_text = " ".join(row["Full-Text"].split()[:6000])
    output = row["abstract"]

    return prompt_template.format(instruction, full_text, output) + tokenizer.eos_token

### ======= モデルの読み込みとデータセットの読み込み ======= ###

# モデルの読み込み
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length = 8192,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)

# データセットの読み込み
train_dataset = pd.read_json("datasets/train.json", dtype = {"id": str})
train_dataset["prompt"] = train_dataset.apply(formatting_prompts_func, axis=1)
train_dataset = Dataset.from_pandas(train_dataset)

valid_dataset = pd.read_json("datasets/valid.json", dtype = {"id": str})
valid_dataset["prompt"] = valid_dataset.apply(formatting_prompts_func, axis=1)
valid_dataset = Dataset.from_pandas(valid_dataset)

### ======= 学習の設定 ======= ###

# LoRA 設定
model = FastLanguageModel.get_peft_model(
    model, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
    r = 8,    
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    max_seq_length = 8192,
    use_rslora = False,
    loftq_config = None,
)

# Train 設定
training_arguments = TrainingArguments(
    bf16 = True,
    group_by_length=True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 8,
    num_train_epochs=5,
    warmup_steps = 60,
    max_steps = 100,
    learning_rate = 2e-4,
    logging_steps = 1,
    output_dir = "checkpoints",
    optim = "adamw_8bit",
    report_to="wandb" 
)

# Trainer の初期化
trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset,
    dataset_text_field = "prompt",
    max_seq_length = 8192,
    tokenizer = tokenizer,
    args = training_arguments,
)

### ======= 学習 ======= ###

wandb.init(project="IIP2")
trainer.train()
wandb.finish()
