import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = [
    "#A11F16",
    "#71B026",
    "#632377",
]
line_styles = [
    (0, (5, 5)),
    (0, (5, 2, 1, 2, 1, 2)),
    (0, (5, 2, 1, 2)),
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
markers = ['^', 'D', '*']
def load_strokepl_data():
    strokepl_files = [
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_2_to_3.csv', '2-3'),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_3_to_4.csv', '3-4'),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_2-3_to_4.csv', '2&3-4')
    ]
    all_data = []
    for file_path, combo in strokepl_files:
        df = pd.read_csv(file_path)
        df['Combo'] = combo
        df['Classifier'] = 'StrokePL'
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def load_gnb_ovsm_data():
    gnb_ovsm_files = [
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_2_3.csv', '2-3'),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_3_4.csv', '3-4'),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_23_4.csv', '2&3-4')
    ]
    all_data = []
    for file_path, combo in gnb_ovsm_files:
        df = pd.read_csv(file_path)
        df['Combo'] = combo
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def calculate_mean_eer(df, posture, pattern_prefix, classifier):
    df_filtered = df[(df['Posture'] == posture) &
                    (df['Pattern'].str.startswith(pattern_prefix)) &
                    (df['Classifier'] == classifier)]
    return df_filtered.groupby('Combo')['EER'].mean().reset_index()

def plot_single_graph(posture, pattern_prefix, output_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strokepl_df = load_strokepl_data()
    gnb_ovsm_df = load_gnb_ovsm_data()
    
    combos = ['2-3', '3-4', '2&3-4']
    x_positions = [1, 2, 3]
    
    classifiers = ['GNB', 'OVSVM', 'StrokePL']
    classifier_colors = [colors[0], colors[1], colors[2]]
    
    for clf_idx, classifier in enumerate(classifiers):
        means = []
        for combo in combos:
            if classifier == 'StrokePL':
                df_filtered = strokepl_df[(strokepl_df['Posture'] == posture) &
                                          (strokepl_df['Pattern'].str.startswith(pattern_prefix)) &
                                          (strokepl_df['Combo'] == combo)]
            else:
                df_filtered = gnb_ovsm_df[(gnb_ovsm_df['Posture'] == posture) &
                                          (gnb_ovsm_df['Pattern'].str.startswith(pattern_prefix)) &
                                          (gnb_ovsm_df['Combo'] == combo) &
                                          (gnb_ovsm_df['Classifier'] == classifier)]
            mean_eer = df_filtered['EER'].mean()
            means.append(mean_eer)
        if markers[clf_idx] == '*':
            ms = 12
        else:
            ms = 8
        ax.plot(x_positions, means, color=classifier_colors[clf_idx], linestyle=line_styles[clf_idx],
                marker=markers[clf_idx], label=classifier, linewidth=2, markersize=ms)
    
    ax.set_xlabel('Training-Testing Combination', fontsize=36)
    ax.set_ylabel('EER', fontsize=36)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(combos, fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.set_ylim(0, 0.6)
    ax.legend(loc='upper right', fontsize=28)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'RQ3-graphs/{output_name}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: RQ3-graphs/{output_name}")

if __name__ == '__main__':
    os.makedirs('RQ3-graphs', exist_ok=True)
    
    pattern_prefixes = ['four', 'three', 'two']
    postures = ['sit', 'walk']
    
    for posture in postures:
        for pattern_prefix in pattern_prefixes:
            output_name = f'eer_{posture}_{pattern_prefix}.png'
            plot_single_graph(posture, pattern_prefix, output_name)