import csv
import matplotlib.pyplot as plt

def read_csv(filepath):
    iterations, losses, accuracies = [], [], []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            iterations.append(int(row['iteration']))
            losses.append(float(row['loss']))
            accuracies.append(float(row['accuracy']))
    return iterations, losses, accuracies

# 读取两个结果文件
dp_iters, dp_losses, dp_accs = read_csv("dp-sst2_999round.csv")
normal_iters, normal_losses, normal_accs = read_csv("sst2_999round.csv")

# 设置颜色和标记大小
dp_color = "#1f77b4"       # 蓝色 (DP)
normal_color = "#2ca02c"   # 绿色 (Normal)
marker_style = 'o'
marker_size = 4  # 小一点的圆点

# ==== 绘制 Accuracy 对比图 ====
plt.figure()
plt.plot(dp_iters, dp_accs, label="DP", color=dp_color, marker=marker_style, markersize=marker_size)
plt.plot(normal_iters, normal_accs, label="Normal", color=normal_color, marker=marker_style, markersize=marker_size)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison (DP vs Normal)")
plt.legend()
plt.grid(True)
plt.savefig("sst2_dp-normal_acc.png")

# ==== 绘制 Loss 对比图 ====
plt.figure()
plt.plot(dp_iters, dp_losses, label="DP", color=dp_color, marker=marker_style, markersize=marker_size)
plt.plot(normal_iters, normal_losses, label="Normal", color=normal_color, marker=marker_style, markersize=marker_size)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Comparison (DP vs Normal)")
plt.legend()
plt.grid(True)
plt.savefig("sst2_dp-normal_loss.png")
