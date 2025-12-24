import streamlit as st
import os
import time

# 关键：设置 HuggingFace 镜像地址 (防止下载连接超时)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 引入 LangChain 组件
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================
# 配置区域 (记得修改 API Key)
# ==========================================
# 🛑 记得把标题改了！
ST_TITLE = "🏥 智能医疗诊断助手 (基于 Qwen-2.5 & RAG)"

# 你的 Key (注意保密)
os.environ["OPENAI_API_KEY"] = "sk-okycixattvhctihwyrnokgeuyylxqxudrykublvsjywwvcdn" 
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ==========================================
# 核心逻辑
# ==========================================
@st.cache_resource
def initialize_rag_system():
    # 1. 加载数据
    if not os.path.exists("clean_medical_knowledge.csv"):
        return None, "请先运行 process_data.py 生成数据文件！"

    print("📄 正在加载医疗数据集...")
    loader = CSVLoader(
        file_path="./clean_medical_knowledge.csv", 
        encoding="utf-8",
        source_column="source"  # 这里指定了 metadata 读取哪一列
    )
    docs = loader.load()

    # ⚠️ 调试用：如果跑得太慢，可以先解除注释下面这行，只取前 1000 条测试
    # docs = docs[:1000]

    # 2. 切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    # 3. 向量化 (使用本地模型)
    print("⬇️ 正在加载本地 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/Xorbits/bge-m3", # 指向你下载好的路径
        model_kwargs={'device': 'cuda'},      # 服务器有显卡就用 cuda，没有就 cpu
        encode_kwargs={'normalize_embeddings': True}
    )

    print("🚀 正在构建向量数据库 (Chroma)...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 4. 构建检索器 (k=3 表示每次找 3 条最相关的)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # 5. 定义 LLM
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.1, # 医疗场景温度要低，保持严谨
        streaming=True
    )

    # 6. 定义 Prompt (改为医疗专家)
    system_prompt = """
    你是一位经验丰富的【三甲医院主治医师】。请基于以下【参考资料】和【对话历史】回答患者的问题。
    
    要求：
    1. 回答必须基于提供的参考资料，严禁编造。
    2. 如果参考资料中没有答案，请直接回答：“抱歉，目前的医疗数据库中没有关于该问题的记录。”
    3. 语气要专业、亲切、富有同理心。

    【参考资料】：
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"), # 👈 这里就是植入记忆的关键
        ("human", "{question}")
    ])

    # 7. 这是一个纯生成链 (Prompt + LLM)
    # 我们把检索步骤移到 UI 层去手动执行，这样就能完美控制流式输出了
    generation_chain = prompt | llm | StrOutputParser()

    # ⚠️ 修改返回值：分别返回 检索器 和 生成链
    return retriever, generation_chain, "系统初始化完成"


# ==========================================
# Streamlit UI 界面逻辑
# ==========================================
st.set_page_config(page_title=ST_TITLE, page_icon="🏥", layout="wide")
st.title(ST_TITLE)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    with st.spinner("正在启动医疗知识引擎..."):
        # 接收两个对象
        retriever, generation_chain, msg = initialize_rag_system()
    
    if retriever and generation_chain: # 判断两个都在
        st.success("✅ 知识库挂载成功")
        st.info(f"🧠 模型: {MODEL_NAME}")
    else:
        st.error(f"❌ 启动失败: {msg}")
        st.stop()

    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.rerun()

# 初始化历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果历史消息里有引用来源，也显示出来
        if "sources" in message:
            with st.expander("📚 参考来源 (Citation)"):
                st.markdown(message["sources"])

# 处理输入
# ==========================================
# 核心交互区域 (包含打字机效果 + 引用显示)
# ==========================================
if prompt := st.chat_input("请描述您的症状或问题..."):
    # 1. 显示用户输入
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # ------------------------------------------
        # 步骤 1: 检索 (Retrieval) - 找资料
        # ------------------------------------------
        status_placeholder = st.empty()
        status_placeholder.markdown("🔍 正在检索医疗数据库...")
        
        # 手动执行检索
        docs = retriever.invoke(prompt)
        
        # 将检索到的文档内容拼接成字符串
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 检索完成后隐藏提示
        status_placeholder.empty()

        # ------------------------------------------
        # 步骤 2: 生成 (Generation) - 流式打字机
        # ------------------------------------------
        response_placeholder = st.empty()
        full_response = ""
        
        # 使用 .stream() 启用流式输出
        # 我们把刚才检索到的 context_text 手动传给链
        try:
            # ✅ 新增代码开始：构建历史记录对象 -------------
            history_buffer = []
            # 遍历历史记录 (排除掉最新的一条用户提问，因为那个会通过 question 参数传入)
            # 注意：st.session_state.messages 此时已经包含了最新的 prompt，所以我们要 [:-1]
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    history_buffer.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_buffer.append(AIMessage(content=msg["content"]))
            # ✅ 新增代码结束 -----------------------------

            # 修改 .stream() 的调用参数
            stream = generation_chain.stream({
                "context": context_text, 
                "question": prompt,
                "chat_history": history_buffer # 👈 把转换好的历史传进去
            })

            for chunk in stream:
                full_response += chunk
                # ▌ 是光标效果，模拟打字
                response_placeholder.markdown(full_response + "▌")
                # 如果本地跑太快看不清，可以取消下面这行的注释
                time.sleep(0.02) 

            # 循环结束，把光标去掉，显示最终完整结果
            response_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"生成时发生错误: {e}")
            full_response = "抱歉，系统生成回答时遇到故障。"

        # ------------------------------------------
        # 步骤 3: 处理引用来源 (Citation)
        # ------------------------------------------
        source_text = ""
        unique_sources = set()
        for doc in docs:
            # 获取 metadata 里的 source
            src = doc.metadata.get('source', '未知来源')
            if src not in unique_sources:
                unique_sources.add(src)
                # 可以在这里做一些美化，比如把 source ID 变成文件名
                source_text += f"- 📄 **证据来源**: `{src}`\n"

        if source_text:
            with st.expander("📚 参考来源 (Citation)"):
                st.markdown(source_text)

    # 4. 存入历史 (包含引用信息)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": source_text # 额外存一个字段
    })