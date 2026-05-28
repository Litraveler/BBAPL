import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = [
    "#A11F16","#71B026", "#632377",
    "#A22017","#E74C3C", "#FF2D6F",
    "#72B127",  "#00B050",  "#00A896",
    "#8E44AD", "#B344C0", "#BE8BD4",
    "#FF00AA",
     "#2C2E35", "#5F6A72", "#A3A9B5",
    "#F25900", "#00C7B1", "#FF9500",
    "#23A4DB", "#FFC100", "#4F61D1", "#FFEE00", "#0075C9",
]

line_styles = [
    (0, (5, 5)),
    (0, (5, 2, 1, 2, 1, 2)),
    (0, (2, 1, 1, 1, 1, 1, 1, 1)),
    (0, (5, 3)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
    
    (0, (3, 1, 1, 1)),
    (0, (2, 1, 1, 1, 1, 1)),
    (0, (1, 2, 1, 2, 1, 2)),
    (0, (2, 1, 2, 1, 2, 1)),
    (0, (3, 1, 3, 1)),
    (0, (1, 1, 3, 1)),
    (0, (4, 1, 1, 1, 1, 1)),
    (0, (5, 1, 1, 1)),
    (0, (4, 2, 1, 2, 1, 2)),
    
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (1, 2, 3, 2))
]

def get_pattern_number(pattern):
    number_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8
    }
    pattern_lower = pattern.lower()
    parts = pattern_lower.split('_')
    if len(parts) >= 2:
        suffix = parts[-1]
        if suffix in number_map:
            return number_map[suffix]
    return 0

def load_gnb_ovsm_data():
    gnb_ovsm_files = [
        ('E:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_1_2.csv', 2),
        ('E:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_1_3.csv', 3),
        ('E:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_1_4.csv', 4)
    ]
    all_data = []
    for file_path, session in gnb_ovsm_files:
        df = pd.read_csv(file_path)
        df['Session'] = session
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def load_strokepl_data():
    strokepl_files = [
        ('E:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_1_to_2.csv', 2),
        ('E:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_1_to_3.csv', 3),
        ('E:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_1_to_4.csv', 4)
    ]
    all_data = []
    for file_path, session in strokepl_files:
        df = pd.read_csv(file_path)
        df['Session'] = session
        df['Classifier'] = 'StrokePL'
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

markers = ['^', 'D', '*']

def plot_single_eer_graph(posture, pattern_prefix, output_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gnb_data = load_gnb_ovsm_data()
    ovsm_data = load_gnb_ovsm_data()
    strokepl_data = load_strokepl_data()
    
    all_pattern_nums = [1, 2, 3, 4, 5, 6, 7, 8]
    
    print(f"\n{'='*60}")
    print(f"Posture: {posture}, Pattern: {pattern_prefix}")
    print(f"{'='*60}")
    
    gnb_color = colors[0]
    ovsm_color = colors[1]
    strokepl_color = colors[2]
    
    for idx, session in enumerate([2, 3, 4]):
        gnb_filtered = gnb_data[(gnb_data['Posture'] == posture) & 
                               (gnb_data['Pattern'].str.startswith(pattern_prefix)) &
                               (gnb_data['Session'] == session) &
                               (gnb_data['Classifier'] == 'GNB')]
        
        gnb_grouped = gnb_filtered.groupby('Pattern')['EER'].mean().reset_index()
        gnb_grouped['PatternNum'] = gnb_grouped['Pattern'].apply(get_pattern_number)
        
        gnb_full = pd.DataFrame({'PatternNum': all_pattern_nums})
        gnb_full = gnb_full.merge(gnb_grouped[['PatternNum', 'EER']], on='PatternNum', how='left')
        
        print(f"\nSession {session} - GNB:")
        print(gnb_full.to_string(index=False))
        if markers[idx] == '*':
            ms = 12
        else:
            ms = 8
        ax.plot(gnb_full['PatternNum'], gnb_full['EER'], 
               color=gnb_color, linestyle=line_styles[idx],
               marker=markers[idx], label=f'Session {session}-GNB', linewidth=2, markersize=ms)
    
    for idx, session in enumerate([2, 3, 4]):
        ovsm_filtered = ovsm_data[(ovsm_data['Posture'] == posture) & 
                                 (ovsm_data['Pattern'].str.startswith(pattern_prefix)) &
                                 (ovsm_data['Session'] == session) &
                                 (ovsm_data['Classifier'] == 'OVSVM')]
        
        ovsm_grouped = ovsm_filtered.groupby('Pattern')['EER'].mean().reset_index()
        ovsm_grouped['PatternNum'] = ovsm_grouped['Pattern'].apply(get_pattern_number)
        
        ovsm_full = pd.DataFrame({'PatternNum': all_pattern_nums})
        ovsm_full = ovsm_full.merge(ovsm_grouped[['PatternNum', 'EER']], on='PatternNum', how='left')
        
        print(f"\nSession {session} - OVSVM:")
        print(ovsm_full.to_string(index=False))
        if markers[idx] == '*':
            ms = 12
        else:
            ms = 8
        ax.plot(ovsm_full['PatternNum'], ovsm_full['EER'], 
               color=ovsm_color, linestyle=line_styles[idx],
               marker=markers[idx], label=f'Session {session}-OVSVM', linewidth=2, markersize=ms)
    
    for idx, session in enumerate([2, 3, 4]):
        strokepl_filtered = strokepl_data[(strokepl_data['Posture'] == posture) & 
                                         (strokepl_data['Pattern'].str.startswith(pattern_prefix)) &
                                         (strokepl_data['Session'] == session)]
        
        strokepl_grouped = strokepl_filtered.groupby('Pattern')['EER'].mean().reset_index()
        strokepl_grouped['PatternNum'] = strokepl_grouped['Pattern'].apply(get_pattern_number)
        
        strokepl_full = pd.DataFrame({'PatternNum': all_pattern_nums})
        strokepl_full = strokepl_full.merge(strokepl_grouped[['PatternNum', 'EER']], on='PatternNum', how='left')
        
        print(f"\nSession {session} - StrokePL:")
        print(strokepl_full.to_string(index=False))
        if markers[idx] == '*':
            ms = 12
        else:
            ms = 8
        ax.plot(strokepl_full['PatternNum'], strokepl_full['EER'], 
               color=strokepl_color, linestyle=line_styles[idx],
               marker=markers[idx], label=f'Session {session}-StrokePL', linewidth=2, markersize=ms)
    
    ax.set_xlabel('Pattern', fontsize=28)
    ax.set_ylabel('EER', fontsize=24)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'RQ2-graphs/{output_name}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: RQ2-graphs/{output_name}")

def plot_legend():
    fig, ax = plt.subplots(figsize=(20, 4))
    
    gnb_color = colors[0]
    ovsm_color = colors[1]
    strokepl_color = colors[2]
    
    for idx, session in enumerate([2, 3, 4]):
        ax.plot([], [], color=gnb_color, linestyle=line_styles[0],
                marker=markers[idx], label=f'GNB-Session {session}', linewidth=2, markersize=12)
    
    for idx, session in enumerate([2, 3, 4]):
        ax.plot([], [], color=ovsm_color, linestyle=line_styles[1],
                marker=markers[idx], label=f'OVSVM-Session {session}', linewidth=2, markersize=12)
    
    for idx, session in enumerate([2, 3, 4]):
        ax.plot([], [], color=strokepl_color, linestyle=line_styles[2],
                marker=markers[idx], label=f'StrokePL-Session {session}', linewidth=2, markersize=12)
    
    # 关键修改：ncol=6，最多6列一行，自动换行
    ax.legend(loc='center', fontsize=24, 
              ncol=6,  # 最多6列一行
              borderpad=1.0, frameon=False)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'RQ2-graphs/legend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: RQ2-graphs/legend.png")

if __name__ == '__main__':
    os.makedirs('RQ2-graphs', exist_ok=True)
    
    pattern_prefixes = ['four', 'three', 'two']
    postures = ['sit', 'walk']
    
    for posture in postures:
        for pattern_prefix in pattern_prefixes:
            output_name = f'eer_{posture}_{pattern_prefix}.png'
            plot_single_eer_graph(posture, pattern_prefix, output_name)
    
    plot_legend()