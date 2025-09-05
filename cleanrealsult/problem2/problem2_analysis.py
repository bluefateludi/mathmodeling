#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：临床检测中，男胎孕妇的BMI是影响胎儿Y染色体浓度的最早达标时间
（即浓度达到4%的最早时间）的主要因素。试对男胎孕妇的BMI进行合理分类，
给出每组的BMI区间和最佳NIPT时点，使得孕妇可能的潜在风险最小，
并分析检测误差对结果的影响。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path):
    """
    加载并预处理数据
    """
    print("正在加载数据...")
    
    # 读取Excel文件
    data = pd.read_excel(file_path)
    
    print(f"原始数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    print("\n数据前5行:")
    print(data.head())
    
    # 检查数据类型
    print("\n数据类型:")
    print(data.dtypes)
    
    # 检查缺失值
    print("\n缺失值统计:")
    print(data.isnull().sum())
    
    return data

def convert_days_to_weeks(data):
    """
    将检测周数列从天数转换为周数
    """
    print("\n正在转换检测周数...")
    
    # 查找检测周数相关的列
    week_columns = [col for col in data.columns if '周' in col or 'week' in col.lower()]
    print(f"找到的周数相关列: {week_columns}")
    
    # 假设检测周数列名为'检测周数'或类似
    if '检测周数' in data.columns:
        print(f"检测周数列原始数据范围: {data['检测周数'].min()} - {data['检测周数'].max()}")
        # 将天数转换为周数
        data['孕周'] = data['检测周数'] / 7
        print(f"转换后孕周范围: {data['孕周'].min():.2f} - {data['孕周'].max():.2f}")
    else:
        # 如果列名不同，尝试其他可能的列名
        for col in week_columns:
            if data[col].dtype in ['int64', 'float64']:
                print(f"使用列 '{col}' 作为检测周数")
                data['孕周'] = data[col] / 7
                break
    
    return data

def analyze_y_chromosome_by_week(data):
    """
    按孕周分析Y染色体浓度变化
    """
    print("\n正在进行孕周分析...")
    
    # 查找Y染色体浓度列
    y_columns = [col for col in data.columns if 'Y' in col or 'y' in col]
    print(f"找到的Y染色体相关列: {y_columns}")
    
    # 假设Y染色体浓度列名包含'Y'或'浓度'
    y_col = None
    for col in data.columns:
        if 'Y' in col and ('浓度' in col or 'concentration' in col.lower()):
            y_col = col
            break
    
    if y_col is None:
        # 如果没找到，使用第一个包含Y的列
        if y_columns:
            y_col = y_columns[0]
        else:
            print("警告：未找到Y染色体浓度列，使用数值列进行分析")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            y_col = numeric_cols[-1]  # 使用最后一个数值列
    
    print(f"使用列 '{y_col}' 作为Y染色体浓度")
    
    # 按孕周分组分析
    weekly_analysis = data.groupby('孕周')[y_col].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]).round(6)
    
    print("\n按孕周的Y染色体浓度统计:")
    print(weekly_analysis.head(10))
    
    return weekly_analysis, y_col

def calculate_probability_above_threshold(mean, std, threshold=0.04):
    """
    计算超过阈值的概率
    """
    if std == 0 or pd.isna(std):
        return 1.0 if mean >= threshold else 0.0
    
    z_score = (threshold - mean) / std
    probability = 1 - stats.norm.cdf(z_score)
    return probability

def find_optimal_timing(data, weekly_analysis, y_col):
    """
    寻找最佳检测时点
    """
    print("\n正在计算最佳检测时点...")
    
    # 计算达到4%浓度的概率
    weekly_analysis['prob_above_4%'] = weekly_analysis.apply(
        lambda row: calculate_probability_above_threshold(row['mean'], row['std']), 
        axis=1
    )
    
    print("\n各孕周达到4%浓度的概率:")
    print(weekly_analysis[['mean', 'std', 'prob_above_4%']].head(15))
    
    # 寻找达到95%概率的最早时点
    optimal_weeks = weekly_analysis[weekly_analysis['prob_above_4%'] >= 0.95]
    
    if not optimal_weeks.empty:
        optimal_week = optimal_weeks.index.min()
        print(f"\n达到4%浓度(95%概率)的最佳检测时点: 第{optimal_week:.1f}周")
    else:
        # 如果没有达到95%概率的，找概率最高的时点
        max_prob_week = weekly_analysis['prob_above_4%'].idxmax()
        max_prob = weekly_analysis.loc[max_prob_week, 'prob_above_4%']
        print(f"\n最高概率时点: 第{max_prob_week:.1f}周，概率: {max_prob:.3f}")
        optimal_week = max_prob_week
    
    return weekly_analysis, optimal_week

def analyze_bmi_groups(data, y_col):
    """
    分析BMI分组
    """
    print("\n正在进行BMI分组分析...")
    
    # 查找BMI列
    bmi_col = None
    for col in data.columns:
        if 'BMI' in col.upper() or 'bmi' in col.lower():
            bmi_col = col
            break
    
    if bmi_col is None:
        print("警告：未找到BMI列")
        return None
    
    print(f"使用列 '{bmi_col}' 作为BMI")
    print(f"BMI数据范围: {data[bmi_col].min():.2f} - {data[bmi_col].max():.2f}")
    
    # 根据WHO标准进行BMI分类
    def classify_bmi(bmi):
        if bmi < 18.5:
            return '偏瘦(<18.5)'
        elif bmi < 25:
            return '正常(18.5-24.9)'
        elif bmi < 30:
            return '超重(25-29.9)'
        else:
            return '肥胖(≥30)'
    
    data['BMI分组'] = data[bmi_col].apply(classify_bmi)
    
    # 统计各BMI组的分布
    bmi_distribution = data['BMI分组'].value_counts()
    print("\nBMI分组分布:")
    print(bmi_distribution)
    
    # 分析各BMI组的最佳时点
    bmi_optimal_timing = {}
    
    for bmi_group in data['BMI分组'].unique():
        group_data = data[data['BMI分组'] == bmi_group]
        
        if len(group_data) < 10:  # 样本量太小的组跳过
            continue
            
        # 按孕周分组分析
        group_weekly = group_data.groupby('孕周')[y_col].agg(['mean', 'std', 'count'])
        
        # 计算概率
        group_weekly['prob_above_4%'] = group_weekly.apply(
            lambda row: calculate_probability_above_threshold(row['mean'], row['std']), 
            axis=1
        )
        
        # 找最佳时点
        optimal_weeks = group_weekly[group_weekly['prob_above_4%'] >= 0.95]
        
        if not optimal_weeks.empty:
            optimal_week = optimal_weeks.index.min()
        else:
            optimal_week = group_weekly['prob_above_4%'].idxmax()
        
        bmi_optimal_timing[bmi_group] = {
            'optimal_week': optimal_week,
            'sample_size': len(group_data),
            'mean_concentration': group_data[y_col].mean(),
            'weekly_analysis': group_weekly
        }
    
    print("\n各BMI组的最佳检测时点:")
    for group, info in bmi_optimal_timing.items():
        print(f"{group}: 第{info['optimal_week']:.1f}周 (样本量: {info['sample_size']})")
    
    return bmi_optimal_timing, bmi_col

def create_visualizations(data, weekly_analysis, bmi_optimal_timing, y_col, output_dir):
    """
    创建可视化图表
    """
    print("\n正在生成可视化图表...")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Y染色体浓度随孕周变化
    axes[0, 0].plot(weekly_analysis.index, weekly_analysis['mean'], 'b-', linewidth=2, label='平均浓度')
    axes[0, 0].fill_between(weekly_analysis.index, 
                           weekly_analysis['mean'] - weekly_analysis['std'],
                           weekly_analysis['mean'] + weekly_analysis['std'],
                           alpha=0.3, label='±1标准差')
    axes[0, 0].axhline(y=0.04, color='r', linestyle='--', label='4%阈值')
    axes[0, 0].set_xlabel('孕周')
    axes[0, 0].set_ylabel('Y染色体浓度')
    axes[0, 0].set_title('Y染色体浓度随孕周变化')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 达到4%概率随孕周变化
    axes[0, 1].plot(weekly_analysis.index, weekly_analysis['prob_above_4%'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95%概率线')
    axes[0, 1].set_xlabel('孕周')
    axes[0, 1].set_ylabel('达到4%浓度的概率')
    axes[0, 1].set_title('达到4%浓度概率随孕周变化')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. BMI分组的Y染色体浓度分布
    if 'BMI分组' in data.columns:
        data.boxplot(column=y_col, by='BMI分组', ax=axes[1, 0])
        axes[1, 0].set_title('不同BMI组的Y染色体浓度分布')
        axes[1, 0].set_xlabel('BMI分组')
        axes[1, 0].set_ylabel('Y染色体浓度')
    
    # 4. BMI组最佳时点对比
    if bmi_optimal_timing:
        groups = list(bmi_optimal_timing.keys())
        optimal_weeks = [bmi_optimal_timing[group]['optimal_week'] for group in groups]
        
        axes[1, 1].bar(groups, optimal_weeks, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        axes[1, 1].set_xlabel('BMI分组')
        axes[1, 1].set_ylabel('最佳检测时点(周)')
        axes[1, 1].set_title('不同BMI组的最佳检测时点')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/problem2_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"图表已保存到: {output_dir}/problem2_analysis_plots.png")

def save_results(weekly_analysis, bmi_optimal_timing, output_dir):
    """
    保存分析结果
    """
    print("\n正在保存分析结果...")
    
    # 保存周分析结果
    weekly_analysis.to_csv(f'{output_dir}/weekly_analysis.csv', encoding='utf-8-sig')
    
    # 保存BMI分组结果
    if bmi_optimal_timing:
        bmi_results = pd.DataFrame({
            'BMI分组': list(bmi_optimal_timing.keys()),
            '最佳检测时点(周)': [info['optimal_week'] for info in bmi_optimal_timing.values()],
            '样本量': [info['sample_size'] for info in bmi_optimal_timing.values()],
            '平均浓度': [info['mean_concentration'] for info in bmi_optimal_timing.values()]
        })
        bmi_results.to_csv(f'{output_dir}/bmi_optimal_timing.csv', index=False, encoding='utf-8-sig')
    
    print(f"结果已保存到: {output_dir}/")

def main():
    """
    主函数
    """
    # 文件路径
    data_file = "d:/Program code/pythonproject/mathmodel/(MAN)final_cleaned_data.xlsx"
    output_dir = "d:/Program code/pythonproject/mathmodel/cleanrealsult/problem2"
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 加载和预处理数据
        data = load_and_preprocess_data(data_file)
        
        # 2. 转换天数为周数
        data = convert_days_to_weeks(data)
        
        # 3. 按孕周分析Y染色体浓度
        weekly_analysis, y_col = analyze_y_chromosome_by_week(data)
        
        # 4. 寻找最佳检测时点
        weekly_analysis, optimal_week = find_optimal_timing(data, weekly_analysis, y_col)
        
        # 5. BMI分组分析
        bmi_optimal_timing, bmi_col = analyze_bmi_groups(data, y_col)
        
        # 6. 创建可视化
        create_visualizations(data, weekly_analysis, bmi_optimal_timing, y_col, output_dir)
        
        # 7. 保存结果
        save_results(weekly_analysis, bmi_optimal_timing, output_dir)
        
        print("\n=== 问题2分析完成 ===")
        print(f"总体最佳检测时点: 第{optimal_week:.1f}周")
        
        if bmi_optimal_timing:
            print("\n各BMI组最佳检测时点:")
            for group, info in bmi_optimal_timing.items():
                print(f"  {group}: 第{info['optimal_week']:.1f}周")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()