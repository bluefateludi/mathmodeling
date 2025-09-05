import pandas as pd
import numpy as np

# 读取Excel文件
file_path = '../../CUMCM2025Problems/C题/附件.xlsx'

try:
    # 尝试读取所有工作表
    excel_file = pd.ExcelFile(file_path)
    print(f"工作表名称: {excel_file.sheet_names}")
    
    # 读取第一个工作表
    df = pd.read_excel(file_path, sheet_name=0)
    
    print(f"\n数据形状: {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n前5行数据:")
    print(df.head())
    
    print(f"\n数据类型:")
    print(df.dtypes)
    
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    
    print(f"\n基本统计信息:")
    print(df.describe())
    
except Exception as e:
    print(f"读取文件时出错: {e}")
    # 尝试读取其他工作表
    try:
        df = pd.read_excel(file_path, sheet_name=1)
        print(f"\n尝试读取第二个工作表...")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"前5行数据:")
        print(df.head())
    except Exception as e2:
        print(f"读取第二个工作表也失败: {e2}")