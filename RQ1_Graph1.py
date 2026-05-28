import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 绘制单会话折线图，共6张，每张四个会话，八种图案
colors = [
    "#A22017","#72B127", "#00C7B1", "#8E44AD", "#B344C0", "#00A896", "#FF00AA",
    "#E74C3C", "#FF2D6F", "#2C2E35", "#5F6A72", "#A3A9B5",
    "#00B050", "#F25900", "#00C7B1", "#FF9500",
    "#23A4DB", "#FFC100", "#4F61D1", "#FFEE00", "#0075C9",
]

line_styles = [
    (0, (5, 5)),    # 虚线
    (0, (5, 3)),    # 短虚线
    (0, (1, 1)),    # 点线
    (0, (5, 2, 1, 2)),    # 点划线
    (0, (5, 2, 1, 2, 1, 2)),    # 双点划线
    (0, (3, 1, 1, 1)),    # 短划线
    (0, (2, 1, 1, 1, 1, 1)),    # 不规则虚线1
    (0, (1, 2, 1, 2, 1, 2)),    # 不规则虚线2
    (0, (2, 1, 2, 1, 2, 1)),    # 不规则虚线3
    (0, (3, 1, 3, 1)),    # 长间隔虚线
    (0, (1, 1, 3, 1)),    # 点加长间隔
    (0, (4, 1, 1, 1, 1, 1)),    # 长划线加短间隔
    (0, (5, 1, 1, 1)),    # 长划线加短点
    (0, (4, 2, 1, 2, 1, 2)),    # 复杂组合1
    (0, (2, 1, 1, 1, 1, 1, 1, 1)),    # 不规则点线
    (0, (3, 1, 1, 1, 1, 1)),    # 短划线加短间隔
    (0, (1, 2, 3, 2))    # 渐变间隔
]

def get_pattern_number(pattern):
    """从Pattern名称中提取数字部分，返回1-8的整数"""
    number_map = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8
    }
    parts = pattern.split('_')
    if len(parts) > 1:
        suffix = parts[-1]
        if suffix in number_map:
            return number_map[suffix]
    return 0

def hex_to_rgba(hex_color, alpha=1.0):
    """将十六进制颜色转换为RGBA格式"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)

def plot_graph(data_list, posture, pattern_prefix, output_dir):
    """绘制单个图"""
    plt.figure(figsize=(10, 6))
    
    for session_idx, data in enumerate(data_list):
        # 过滤数据：特定的posture和pattern前缀
        filtered_data = data[(data['Posture'] == posture) & (data['Pattern'].str.startswith(pattern_prefix))].copy()
        
        if filtered_data.empty:
            continue
        
        # 添加pattern数字列
        filtered_data['PatternNum'] = filtered_data['Pattern'].apply(get_pattern_number)
        
        # 按PatternNum分组计算统计量
        grouped = filtered_data.groupby('PatternNum')['EER'].agg(['mean', 'min', 'max']).reset_index()
        
        # 按PatternNum排序
        grouped = grouped.sort_values('PatternNum')
        
        x = grouped['PatternNum'].values
        mean_eer = grouped['mean'].values
        
        color = colors[session_idx % len(colors)]
        linestyle = line_styles[session_idx % len(line_styles)]
        
        plt.plot(x, mean_eer, color=color, linestyle=linestyle, linewidth=2, 
                 marker='o', markersize=6, label=f'Session {session_idx + 1}')
    
    plt.xlabel('Pattern', fontsize=28)
    plt.ylabel('EER', fontsize=28)
    plt.legend(fontsize=24, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 9), fontsize=18)
    plt.ylim(0, 1)
    plt.yticks(fontsize=24)
    
    # 保存图片
    output_file = os.path.join(output_dir, f'{posture}_{pattern_prefix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 创建输出目录
    output_dir = 'RQ1-graphs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取四个CSV文件
    csv_files = [
        'single_timeperiod_eer_results_FIX_1.csv',
        'single_timeperiod_eer_results_FIX_2.csv',
        'single_timeperiod_eer_results_FIX_3.csv',
        'single_timeperiod_eer_results_FIX_4.csv'
    ]
    
    data_list = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            data_list.append(df)
            print(f"读取文件: {csv_file}, 行数: {len(df)}")
        else:
            print(f"警告: 文件不存在: {csv_file}")
    
    if not data_list:
        print("错误: 没有找到任何CSV文件")
        return
    
    # 定义要绘制的组合
    postures = ['sit', 'walk']
    pattern_prefixes = ['four', 'three', 'two']
    
    # 绘制所有组合的图
    for posture in postures:
        for pattern_prefix in pattern_prefixes:
            plot_graph(data_list, posture, pattern_prefix, output_dir)
            print(f"已生成图: {posture}_{pattern_prefix}.png")
    
    print(f"\n所有图片已保存到目录: {output_dir}")

if __name__ == '__main__':
    main()