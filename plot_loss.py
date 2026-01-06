import json
import matplotlib.pyplot as plt
import os

# 1. 找到 trainer_state.json 的路径 (请根据你的 checkpoint 文件夹名称修改)
# 通常在 results/checkpoint-XXX/trainer_state.json 或者是保存后的目录
log_file = "./results/trainer_state.json" 

if not os.path.exists(log_file):
    # 如果根目录没有，尝试去最新的 checkpoint 文件夹找
    checkpoints = [d for d in os.listdir("./results") if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        log_file = f"./results/{latest_checkpoint}/trainer_state.json"

if os.path.exists(log_file):
    with open(log_file, "r") as f:
        data = json.load(f)

    # 2. 提取步数和 Loss
    steps = []
    loss_values = []
    for entry in data["log_history"]:
        if "loss" in entry:
            steps.append(entry["step"])
            loss_values.append(entry["loss"])

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_values, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Medical SFT Training Loss')
    plt.legend()
    plt.grid(True)
    
    # 4. 保存图片
    plt.savefig("loss_curve.png")
    print("✅ Loss 曲线已保存至 loss_curve.png")
else:
    print(f"❌ 未找到日志文件: {log_file}")