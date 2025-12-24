import pandas as pd
import json
import os

# =================配置区域=================
QUESTION_FILE = './data/question.csv'
ANSWER_FILE = './data/answer.csv'
OUTPUT_FILE = 'test_dataset.json'
SAMPLE_SIZE = 20  # 只需要 20 个题来做评估演示
# ==========================================

def create_test_set():
    print("🚀 正在生成测试集...")
    
    # 1. 读取原始数据 (处理编码问题)
    try:
        df_q = pd.read_csv(QUESTION_FILE, names=['qid', 'content'], encoding='utf-8')
        df_a = pd.read_csv(ANSWER_FILE, names=['aid', 'qid', 'content'], encoding='utf-8')
    except:
        df_q = pd.read_csv(QUESTION_FILE, names=['qid', 'content'], encoding='gbk')
        df_a = pd.read_csv(ANSWER_FILE, names=['aid', 'qid', 'content'], encoding='gbk')

    # 2. 合并
    merged = pd.merge(df_a, df_q, on='qid', suffixes=('_ans', '_ask'))
    
    # 3. 过滤短回答 (我们要高质量的长答案作为标准)
    merged = merged[merged['content_ans'].str.len() > 20]
    
    # 4. 随机抽取 20 条
    # random_state=999 保证抽出来的和之前建库的大概率不一样
    sample = merged.sample(n=SAMPLE_SIZE, random_state=999)

    # 5. 格式化为 Ragas 需要的列表格式
    test_data = []
    for index, row in sample.iterrows():
        test_data.append({
            "question": row['content_ask'],
            "ground_truth": row['content_ans'] # 标准答案
        })

    # 6. 保存为 JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 测试集生成完毕！已保存为 {OUTPUT_FILE}")
    print(f"预览第一条:\n问题: {test_data[0]['question']}\n答案: {test_data[0]['ground_truth'][:50]}...")

if __name__ == "__main__":
    create_test_set()