import pandas as pd
import os

# 定义两个文件的路径
file_path1 = r"E:\论文撰写记录\投稿\StrokePL\Codes\datas\padding_data.csv"
file_path2 = r"E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas\padding_data.csv"

# 定义合并后文件的保存路径（可根据需要修改）
output_path = r"E:\论文撰写记录\投稿\StrokePL\Codes\merged_padding_data.csv"

def merge_csv_files(file1, file2, output_file):
    """
    合并两个列名相同但顺序可能不同的CSV文件，先统一列顺序再合并
    
    参数:
        file1: 第一个CSV文件路径（作为列顺序的基准）
        file2: 第二个CSV文件路径
        output_file: 合并后文件的保存路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file1):
            raise FileNotFoundError(f"文件不存在: {file1}")
        if not os.path.exists(file2):
            raise FileNotFoundError(f"文件不存在: {file2}")
        
        # 读取两个CSV文件
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # 第一步：验证列名是否完全一致（不考虑顺序）
        set1 = set(df1.columns)
        set2 = set(df2.columns)
        if set1 != set2:
            # 找出差异的列，方便排查问题
            missing_in_df2 = set1 - set2
            missing_in_df1 = set2 - set1
            error_msg = "两个CSV文件的列名不一致！\n"
            if missing_in_df2:
                error_msg += f"df2缺少列: {missing_in_df2}\n"
            if missing_in_df1:
                error_msg += f"df1缺少列: {missing_in_df1}"
            raise ValueError(error_msg)
        
        # 第二步：将df2的列顺序调整为和df1完全一致
        df2 = df2[df1.columns]
        print(f"已将第二个文件的列顺序调整为与第一个文件一致，基准列顺序：\n{list(df1.columns)}")
        
        # 第三步：合并数据（行拼接）
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # 保存合并后的数据
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 输出合并信息
        print(f"\n合并完成！")
        print(f"第一个文件行数: {len(df1)}")
        print(f"第二个文件行数: {len(df2)}")
        print(f"合并后总行数: {len(merged_df)}")
        print(f"合并后的文件已保存至: {output_file}")
        
    except Exception as e:
        print(f"合并过程中出现错误: {str(e)}")

# 执行合并操作
merge_csv_files(file_path1, file_path2, output_path)