import pandas as pd
import json

# 读取原始数据
df_q = pd.read_csv('./data/question.csv', names=['qid', 'content'], encoding='utf-8')
df_a = pd.read_csv('./data/answer.csv', names=['aid', 'qid', 'content'], encoding='utf-8')

# 合并
merged = pd.merge(df_a, df_q, on='qid', suffixes=('_ans', '_ask'))

# 过滤短回答
merged = merged[merged['content_ans'].str.len() > 10]

# =======================================================
# 👇 新增代码：只随机抽取 10,000 条
# =======================================================
if len(merged) > 10000:
    merged = merged.sample(n=10000, random_state=42)
    print(f"✂️ 已随机采样 10,000 条数据用于微调")
else:
    print(f"⚠️ 数据不足 10,000 条，将使用全部 {len(merged)} 条")
# =======================================================

# 转换为 Qwen/LLaMA 常见的指令微调格式
# 格式: {"instruction": "...", "input": "", "output": "..."}
sft_data = []
for _, row in merged.iterrows():
    sft_data.append({
        "instruction": "你是一名专业的医生。请根据患者的描述回答问题，回答要专业、亲切。",
        "input": row['content_ask'],
        "output": row['content_ans']
    })

# 保存为 JSONL
with open('medical_sft_data.jsonl', 'w', encoding='utf-8') as f:
    for item in sft_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ 微调数据已生成，共 {len(sft_data)} 条")