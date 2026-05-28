import pandas as pd

# 读取CSV文件
file_path = 'E7_results_entropy/entropy.csv'
df = pd.read_csv(file_path)

# 按posture分组并计算所需列的均值
grouped = df.groupby('posture')[
    ['beta_entropy_3', 'beta_entropy_6', 'beta_entropy_10',
     'alfa_guess_work_entropy_5', 'alfa_guess_work_entropy_2',
     'upper_bound']
].mean().reset_index()

# 重命名列名使其更清晰（可选）
grouped.columns = ['posture',
                  'mean_beta_entropy_3',
                  'mean_beta_entropy_6',
                  'mean_beta_entropy_10',
                  'mean_alfa_guess_work_entropy_5',
                  'mean_alfa_guess_work_entropy_2',
                  'mean_upper_bound']

# 打印结果
print(grouped)

# 可选：将结果保存到新CSV文件
grouped.to_csv('E7_results_entropy/posture_group_means.csv', index=False)