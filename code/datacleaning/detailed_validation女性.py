#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('../../Womenclean.xlsx')

print('=== 详细缺失值分析 ===')
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
print('各字段缺失值统计:')
for col, count in missing_summary.items():
    print(f'{col}: {count} ({count/len(df)*100:.2f}%)')

print('\n=== 检测孕周字段详细分析 ===')
print(f'检测孕周范围: {df["检测孕周"].min():.2f} - {df["检测孕周"].max():.2f} 周')
print(f'异常值检查 (< 10周 或 > 42周):')
abnormal = df[(df['检测孕周'] < 10) | (df['检测孕周'] > 42)]
print(f'异常值数量: {len(abnormal)}')
if len(abnormal) > 0:
    print(abnormal[['序号', '检测孕周']].head())

print('\n=== 数据类型检查 ===')
print(df.dtypes.value_counts())

print('\n=== 关键字段数据分布 ===')
print('\n年龄分布:')
print(df['年龄'].describe())
print('\n身高分布:')
print(df['身高'].describe())
print('\n体重分布:')
print(df['体重'].describe())
print('\nBMI分布:')
print(df['孕妇BMI'].describe())

print('\n=== 分类字段分析 ===')
print('\nIVF妊娠分布:')
print(df['IVF妊娠'].value_counts())
print('\n胎儿是否健康分布:')
print(df['胎儿是否健康'].value_counts())
print('\n染色体的非整倍体分布:')
print(df['染色体的非整倍体'].value_counts(dropna=False))

print('\n=== 数据质量总结 ===')
total_cells = df.shape[0] * df.shape[1]
missing_cells = df.isnull().sum().sum()
completeness = (total_cells - missing_cells) / total_cells * 100
print(f'总记录数: {df.shape[0]}')
print(f'总字段数: {df.shape[1]}')
print(f'总单元格数: {total_cells}')
print(f'缺失单元格数: {missing_cells}')
print(f'数据完整性: {completeness:.2f}%')
print(f'主要缺失字段: 染色体的非整倍体 ({missing_summary.iloc[0] if len(missing_summary) > 0 else 0} 个缺失值)')