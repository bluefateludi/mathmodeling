#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# 读取最终清洗后的数据
df = pd.read_excel('../../Womenclean.xlsx')

print("=== 最终清洗后数据验证 ===")
print(f'数据形状: {df.shape}')
print(f'缺失值总数: {df.isnull().sum().sum()}')
print(f'数据完整性: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.2f}%')

print("\n=== 检测孕周字段统计 ===")
print(df['检测孕周'].describe())
print(f'检测孕周数据类型: {df["检测孕周"].dtype}')
print(f'检测孕周缺失值: {df["检测孕周"].isnull().sum()}')

print("\n=== 染色体的非整倍体字段 ===")
print(f'唯一值: {df["染色体的非整倍体"].unique()}')
print(f'缺失值: {df["染色体的非整倍体"].isnull().sum()}')

print("\n=== 其他关键字段缺失值统计 ===")
key_columns = ['年龄', '身高', '体重', '孕妇BMI', 'IVF妊娠', '胎儿是否健康']
for col in key_columns:
    missing = df[col].isnull().sum()
    print(f'{col}: {missing} 个缺失值')

print("\n=== 数据清洗效果对比 ===")
# 读取原始数据进行对比
original = pd.read_excel('../../CUMCM2025Problems/C题/附件.xlsx')
print(f'原始数据: {original.shape[0]} 行, 缺失值: {original.isnull().sum().sum()}')
print(f'清洗后数据: {df.shape[0]} 行, 缺失值: {df.isnull().sum().sum()}')
print(f'删除记录数: {original.shape[0] - df.shape[0]}')
print(f'缺失值减少: {original.isnull().sum().sum() - df.isnull().sum().sum()}')