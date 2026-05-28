import numpy as np
import matplotlib.pyplot as plt

# 定义函数 H(s)
def H(s):
    term1 = ((1 + s) / (2 * s)) * np.log((1 + s) / (2 * s))
    term2 = ((1 - s) / (2 * s)) * np.log((1 - s) / (2 * s))
    return term1 - term2

# 定义 s 的范围，从 0 到 0.1，间隔为 0.01
s = np.arange(0.01, 0.11, 0.01)  # 从 0.01 开始，到 0.1 结束，步长为 0.01

# 计算 H(s) 的值
H_values = H(s)

# 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(s, H_values, label='H(s)', color='blue', marker='o', linestyle='-')
plt.xlabel('s', fontsize=14)
plt.ylabel('H(s)', fontsize=14)
plt.title('Entropy Function H(s) for 0 < s ≤ 0.1', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(0, 0.11)  # 设置横坐标范围为 0 到 0.11，稍微超出一点以便显示最后一个点
plt.xticks(np.arange(0, 0.11, 0.01))  # 设置横坐标刻度为 0.01 的间隔
plt.show()