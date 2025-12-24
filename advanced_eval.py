import json
import pandas as pd
import time
import re
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from exp import initialize_rag_system

# ================= 配置区域 =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OPENAI_API_KEY"] = "sk-okycixattvhctihwyrnokgeuyylxqxudrykublvsjywwvcdn" 
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"
# ===========================================

# 🔥 核心升级：多维度评分 Prompt
# 我们要求 AI 从三个独立维度打分，并输出 JSON
# 🔥 核心升级：CoT (思维链) 评分模板
ADVANCED_EVAL_TEMPLATE = """
你是一位极其严格的 NLP 评估专家。请基于参考资料和标准答案，对考生回答进行“找茬”式评分。

【参考资料 (Context)】：
{context}

【标准答案 (Ground Truth)】：
{ground_truth}

【考生回答 (Answer)】：
{answer}

---
请按照以下步骤思考（不要跳过！）：
1. 检查 **准确性**：考生回答是否遗漏了标准答案里的关键点？(遗漏了就扣分)
2. 检查 **忠实度**：考生回答里有没有参考资料里没提到的废话？(有废话必须扣分，哪怕是对的也要扣！)
3. 检查 **引用**：考生是否充分利用了资料？

最后输出 JSON。

请严格按照以下 JSON 格式输出（分数必须是 0.0, 0.3, 0.5, 0.8, 1.0 中的一个，以此拉开差距）：
{{
    "reasoning": "简短的一句话，指出具体哪里扣分了",
    "accuracy": 0.x,
    "faithfulness": 0.x,
    "citation_f1": 0.x
}}
"""

def advanced_evaluate():
    print("🚀 初始化 RAG 系统 (高级模式)...")
    retriever, generation_chain, _ = initialize_rag_system()
    
    # ⚠️ 使用 temperature=0.0，让模型输出 JSON 更稳定
    evaluator_llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct", temperature=0.0)
    eval_chain = ChatPromptTemplate.from_template(ADVANCED_EVAL_TEMPLATE) | evaluator_llm | StrOutputParser()

    print("📂 读取测试集 test_dataset.json ...")
    with open('test_dataset.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # ⚠️ 修改文件名，避免和之前的混淆
    output_file = "advanced_evaluation.xlsx"
    results = []
    
    # === 断点续传逻辑 ===
    if os.path.exists(output_file):
        print("🔄 检测到上次运行的记录，正在尝试读取...")
        try:
            existing_df = pd.read_excel(output_file)
            # 检查文件是否包含新指标列，如果是旧格式则不读取
            if 'accuracy' in existing_df.columns:
                results = existing_df.to_dict('records')
                print(f"✅ 已跳过前 {len(results)} 个已完成的问题")
            else:
                print("⚠️ 检测到旧格式表格，将重新开始生成...")
                results = []
        except:
            print("⚠️ 读取失败，重新开始")
            results = []

    print(f"⚡ 开始硬核评估 (共 {len(test_data)} 题)...")
    
    start_index = len(results)
    
    for i in range(start_index, len(test_data)):
        item = test_data[i]
        q = item['question']
        gt = item['ground_truth']
        
        print(f"\n-------- 第 {i+1}/{len(test_data)} 题 --------")
        print(f"❓ 问题: {q}")
        
        try:
            # 1. 检索 (Context 非常重要！)
            docs = retriever.invoke(q)
            # 给文档加个序号，方便 LLM 识别
            context_text = "\n".join([f"[{j+1}] {d.page_content}" for j, d in enumerate(docs)])
            
            # 2. 生成回答
            print("🤖 正在生成回答...")
            response = generation_chain.invoke({
            "context": context_text, 
            "question": q,
            "chat_history": [] 
            })
            print(f"💬 回答预览: {response[:20]}...")
            
            # 🛑 休息 20 秒 (保持你的安全设置)
            print("⏳ 生成完毕，休息 20 秒...")
            time.sleep(20) 
            
            # 3. LLM 三维判卷
            print("👨‍🏫 正在进行多维度评分...")
            eval_result_str = eval_chain.invoke({
                "context": context_text,
                "ground_truth": gt,
                "answer": response
            })
            
            # 4. 解析 JSON (清洗数据)
            # 有时候模型会加 ```json ... ```，需要去掉
            clean_json = eval_result_str.replace("```json", "").replace("```", "").strip()
            
            try:
                scores = json.loads(clean_json)
            except json.JSONDecodeError:
                # 万一解析失败，给个保底分，并记录错误
                print(f"⚠️ JSON 解析失败，原始返回: {clean_json}")
                scores = {"accuracy": 0.5, "faithfulness": 0.5, "citation_f1": 0.5, "reason": "解析失败"}
            
            print(f"📊 结果: accuracy{scores.get('accuracy')} / faithfulness{scores.get('faithfulness')} / citation_f1{scores.get('citation_f1')}")
            
            # 5. 存入结果
            results.append({
                "question": q,
                "ground_truth": gt,
                "answer": response,
                "contexts": context_text, # 把参考资料也存下来，显得专业
                "accuracy": scores.get('accuracy', 0),
                "citation_f1": scores.get('citation_f1', 0),
                "faithfulness": scores.get('faithfulness', 0),
                # 自动计算幻觉率
                "hallucination_rate": 1.0 - float(scores.get('faithfulness', 0)), 
                "reason": scores.get('reason', '')
            })
            
            # 💾 实时保存
            pd.DataFrame(results).to_excel(output_file, index=False)
            print("💾 进度已保存")
            
            # 🛑 再次休息 20 秒
            print("⏳ 评分完毕，休息 20 秒...")
            time.sleep(20)

        except Exception as e:
            print(f"❌ 本题出错: {e}")
            # 出错后的长休息
            time.sleep(60)

    print(f"\n🎉 硬核评估完成！请查看 {output_file}")

if __name__ == "__main__":
    advanced_evaluate()