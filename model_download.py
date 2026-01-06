# download.py
import os
from modelscope import snapshot_download

# 1. 确保目标文件夹存在
save_dir = "./models"
os.makedirs(save_dir, exist_ok=True)

# 2. 下载大模型Qwen2.5-7B-Instruct到当前目录下的 models 文件夹
print("🚀 正在下载 Qwen2.5-7B-Instruct...")
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='./models')
print(f"✅ 下载完成，路径: {model_dir}")

print("🚀 正在从魔搭社区下载 BGE-M3 模型...")
print("这可能需要 1-2 分钟，请耐心等待...")

# 2. 下载Embedding模型bge-m3到当前目录下的 models 文件夹
model_dir = snapshot_download(
    'Xorbits/bge-m3', 
    cache_dir=save_dir, 
    revision='master'
)

print(f"✅ 下载成功！模型已保存在: {model_dir}")