import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# 1. 配置路径和参数
model_name = "./models/Qwen/Qwen2.5-7B-Instruct" # 或者本地路径
new_model_name = "Qwen2.5-Medical-LoRA"
dataset_file = "medical_sft_data.jsonl"

# 2. 加载量化配置 (4-bit QLoRA，显存占用极低)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 3. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # 修复 pad token 问题

# 4. 配置 LoRA
peft_config = LoraConfig(
    r=16,       # LoRA 秩，越大参数越多
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 针对 Qwen 的全模块微调
)

# 5. 加载数据集
dataset = load_dataset("json", data_files=dataset_file, split="train")

# 6. 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,          # 演示用 1 epoch，实际建议 3-5
    # ============================================================
    # 👇 必须修改这 3 行来拯救显存
    # ============================================================
    per_device_train_batch_size=1,       # ❌ 原来是 4 -> 改为 1 (最关键)
    gradient_accumulation_steps=4,       # ❌ 原来是 1 -> 改为 4 (保持总批次不变)
    gradient_checkpointing=True,         # ✅ 必须新增这一行！(用时间换空间，省显存神器)
    # ============================================================
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    logging_steps=25,
    report_to="tensorboard",
    save_steps=100,
    optim="paged_adamw_32bit",   # 节省显存的关键优化器
)

# ==========================================
# 7. 开始训练 (修正版)
# ==========================================

# 1. 定义格式化函数：把数据拼成 Qwen 的对话格式
def formatting_prompts_func(example):
    output_texts = []
    # 遍历数据集中的每一条
    for i in range(len(example['instruction'])):
        # 构建 ChatML 格式: <|im_start|>role...<|im_end|>
        text = (
            f"<|im_start|>system\n{example['instruction'][i]}<|im_end|>\n"
            f"<|im_start|>user\n{example['input'][i]}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output'][i]}<|im_end|>"
        )
        output_texts.append(text)
    return output_texts

# 2. 初始化 Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func, # ✅ 使用自定义的拼接函数
    # dataset_text_field="output",           # ❌ 删掉这行，否则会冲突
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512,
)

print("🚀 开始微调...")
trainer.train()

# 8. 保存微调后的适配器 (Adapter)
trainer.model.save_pretrained(new_model_name)
print(f"✅ 微调完成！模型已保存至 {new_model_name}")