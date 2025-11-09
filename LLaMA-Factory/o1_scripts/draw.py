import json
import matplotlib.pyplot as plt

attr = "loss"#"grad_norm"#
data = json.load(open("./saves/math/Marco-loft-iter0/trainer_state.json","r"))["log_history"][:-1][0:90]
print(len(data))

steps = [entry["step"] for entry in data]
grad_norms = [entry[attr] for entry in data]
# grad_norms = [entry["learning_rate"] for entry in data]

# 绘制图形
plt.figure(figsize=(8, 5))
plt.plot(steps, grad_norms, marker='o', linestyle='-', color='b', markersize=1, label='Grad Norm')
plt.title(f'{attr} by Step')
plt.xlabel('Step')
plt.ylabel(attr)
plt.grid()
plt.legend()

# 保存图像到文件
plt.savefig(f'{attr}_by_step.png', dpi=100, bbox_inches='tight')