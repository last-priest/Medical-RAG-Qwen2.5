import pandas as pd
import os

# =================配置区域=================
# 原始文件路径 (根据你的截图修改)
QUESTION_FILE = './data/question.csv'
ANSWER_FILE = './data/answer.csv'

# 输出文件路径
OUTPUT_FILE = './data/clean_medical_knowledge.csv'

# 采样数量 (作业要求 > 5k，我们取 10k 比较稳)
SAMPLE_SIZE = 10000
# ==========================================

def process_cmedqa():
    print("🚀 开始读取数据...")
    
    # 1. 读取 CSV (cMedQA 通常没有表头，我们需要手动指定 names)
    # 根据 cMedQA2 的常见格式：
    # question.csv: [question_id, content]
    # answer.csv: [answer_id, question_id, content]
    
    try:
        df_q = pd.read_csv(QUESTION_FILE, names=['qid', 'content'], encoding='utf-8')
        df_a = pd.read_csv(ANSWER_FILE, names=['aid', 'qid', 'content'], encoding='utf-8')
    except UnicodeDecodeError:
        # 如果 utf-8 报错，尝试 gbk (中文常见编码)
        print("⚠️ UTF-8 读取失败，尝试 GBK 编码...")
        df_q = pd.read_csv(QUESTION_FILE, names=['qid', 'content'], encoding='gbk')
        df_a = pd.read_csv(ANSWER_FILE, names=['aid', 'qid', 'content'], encoding='gbk')

    print(f"📊 原始数据统计: 问题 {len(df_q)} 条, 回答 {len(df_a)} 条")

    # 2. 数据合并 (Left Join)
    # 我们把“问题”合并到“回答”上，通过 'qid' 关联
    print("🔗 正在合并问题和答案...")
    merged_df = pd.merge(df_a, df_q, on='qid', suffixes=('_ans', '_ask'))
    
    # merged_df 现在包含: aid, qid, content_ans (回答), content_ask (问题)

    # 3. 过滤过短的回答 (比如 "好的", "谢谢") -> 这种对 RAG 没用
    merged_df = merged_df[merged_df['content_ans'].str.len() > 10]

    # 4. 格式化为 RAG 可用的文本
    # 格式： "问题：xxxxx \n 医生回答：xxxxx"
    # 这样 RAG 检索时既能匹配到问题的关键词，又能提供答案
    merged_df['rag_content'] = (
        "【患者提问】：" + merged_df['content_ask'] + "\n" +
        "【医生回答】：" + merged_df['content_ans']
    )

    # 5. 随机采样 (完成作业要求)
    if len(merged_df) > SAMPLE_SIZE:
        sampled_df = merged_df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"✂️ 数据量过大，已随机采样 {SAMPLE_SIZE} 条用于作业。")
    else:
        sampled_df = merged_df
        print(f"✅ 数据量符合要求 ({len(merged_df)} 条)。")

    # 6. 保存清洗后的数据
    # 我们只保留 'source' (来源用于引用) 和 'rag_content' (用于检索)
    final_df = pd.DataFrame({
        'content': sampled_df['rag_content'],
        'source': 'cMedQA2_ID_' + sampled_df['qid'] # 模拟一个引用来源 ID
    })
    
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"🎉 处理完成！清洗后的数据已保存为: {OUTPUT_FILE}")
    print("前 3 条数据示例：")
    print(final_df.head(3))

if __name__ == "__main__":
    if not os.path.exists(QUESTION_FILE) or not os.path.exists(ANSWER_FILE):
        print(f"❌ 错误：请确保 {QUESTION_FILE} 和 {ANSWER_FILE} 在当前目录下！")
    else:
        process_cmedqa()