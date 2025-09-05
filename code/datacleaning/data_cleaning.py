#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT数据清洗脚本
根据NIPT问题解决方案进行数据清洗处理
包括：缺失值处理、格式错误修正、重复记录去除
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NIPTDataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.original_data = None
        self.cleaned_data = None
        self.cleaning_report = {}
        
    def load_data(self):
        """加载原始数据"""
        try:
            self.original_data = pd.read_excel(self.file_path)
            print(f"成功加载数据，原始数据形状: {self.original_data.shape}")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def analyze_data_quality(self):
        """分析数据质量"""
        if self.original_data is None:
            print("请先加载数据")
            return
        
        print("\n=== 数据质量分析 ===")
        print(f"数据形状: {self.original_data.shape}")
        print(f"列名: {list(self.original_data.columns)}")
        
        # 缺失值分析
        missing_values = self.original_data.isnull().sum()
        missing_percent = (missing_values / len(self.original_data)) * 100
        
        missing_df = pd.DataFrame({
            '缺失数量': missing_values,
            '缺失百分比': missing_percent
        })
        missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
        
        print("\n缺失值统计:")
        print(missing_df)
        
        # 重复值分析
        duplicates = self.original_data.duplicated().sum()
        print(f"\n重复记录数量: {duplicates}")
        
        # 数据类型分析
        print("\n数据类型:")
        print(self.original_data.dtypes)
        
        return missing_df, duplicates
    
    def clean_missing_values(self):
        """处理缺失值"""
        print("\n=== 处理缺失值 ===")
        
        # 复制原始数据
        self.cleaned_data = self.original_data.copy()
        
        # 记录清洗前的缺失值情况
        before_missing = self.cleaned_data.isnull().sum().sum()
        
        # 1. 处理'末次月经'字段的缺失值
        # 如果缺失值较少，可以删除这些记录
        lmp_missing = self.cleaned_data['末次月经'].isnull().sum()
        if lmp_missing > 0:
            print(f"删除'末次月经'缺失的 {lmp_missing} 条记录")
            self.cleaned_data = self.cleaned_data.dropna(subset=['末次月经'])
        
        # 2. 处理'染色体的非整倍体'字段
        # 这个字段缺失值很多(956个)，可能是正常情况（表示无异常）
        # 将缺失值填充为'正常'或0
        aneuploidy_missing = self.cleaned_data['染色体的非整倍体'].isnull().sum()
        if aneuploidy_missing > 0:
            print(f"将'染色体的非整倍体'字段的 {aneuploidy_missing} 个缺失值填充为'正常'")
            self.cleaned_data['染色体的非整倍体'].fillna('正常', inplace=True)
        
        # 3. 检查其他数值字段的异常值
        numeric_columns = self.cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['年龄', '身高', '体重', '检测孕周', '孕妇BMI']:
                # 检查是否有不合理的值
                if col == '年龄':
                    # 年龄应该在合理范围内
                    invalid_age = (self.cleaned_data[col] < 15) | (self.cleaned_data[col] > 50)
                    if invalid_age.sum() > 0:
                        print(f"发现 {invalid_age.sum()} 个异常年龄值")
                        
                elif col == '身高':
                    # 身高应该在合理范围内(cm)
                    invalid_height = (self.cleaned_data[col] < 140) | (self.cleaned_data[col] > 200)
                    if invalid_height.sum() > 0:
                        print(f"发现 {invalid_height.sum()} 个异常身高值")
                        
                elif col == '体重':
                    # 体重应该在合理范围内(kg)
                    invalid_weight = (self.cleaned_data[col] < 35) | (self.cleaned_data[col] > 150)
                    if invalid_weight.sum() > 0:
                        print(f"发现 {invalid_weight.sum()} 个异常体重值")
                        
                elif col == '检测孕周':
                    # 检测孕周现在是天数，应该在合理范围内(70-294天，对应10-42周)
                    invalid_weeks = (self.cleaned_data[col] < 70) | (self.cleaned_data[col] > 294)
                    if invalid_weeks.sum() > 0:
                        print(f"发现 {invalid_weeks.sum()} 个异常孕周天数值")
                        # 可以选择删除异常值或设为NaN
                        # self.cleaned_data.loc[invalid_weeks, col] = np.nan
                        
                elif col == '孕妇BMI':
                    # BMI应该在合理范围内
                    invalid_bmi = (self.cleaned_data[col] < 15) | (self.cleaned_data[col] > 45)
                    if invalid_bmi.sum() > 0:
                        print(f"发现 {invalid_bmi.sum()} 个异常BMI值")
        
        # 记录清洗后的缺失值情况
        after_missing = self.cleaned_data.isnull().sum().sum()
        
        self.cleaning_report['missing_values'] = {
            'before': before_missing,
            'after': after_missing,
            'removed': before_missing - after_missing
        }
        
        print(f"缺失值处理完成: {before_missing} -> {after_missing}")
    
    def remove_duplicates(self):
        """去除重复记录"""
        print("\n=== 去除重复记录 ===")
        
        before_count = len(self.cleaned_data)
        duplicates_count = self.cleaned_data.duplicated().sum()
        
        if duplicates_count > 0:
            print(f"发现 {duplicates_count} 条重复记录")
            self.cleaned_data = self.cleaned_data.drop_duplicates()
            after_count = len(self.cleaned_data)
            print(f"删除重复记录后: {before_count} -> {after_count}")
        else:
            print("未发现重复记录")
            after_count = before_count
        
        self.cleaning_report['duplicates'] = {
            'before_count': before_count,
            'duplicates_found': duplicates_count,
            'after_count': after_count
        }
    
    def validate_data_formats(self):
        """验证和修正数据格式"""
        print("\n=== 验证数据格式 ===")
        
        format_issues = []
        
        # 1. 检查日期格式
        date_columns = ['末次月经', '检测日期']
        for col in date_columns:
            if col in self.cleaned_data.columns:
                try:
                    # 尝试转换为日期格式
                    self.cleaned_data[col] = pd.to_datetime(self.cleaned_data[col], errors='coerce')
                    invalid_dates = self.cleaned_data[col].isnull().sum()
                    if invalid_dates > 0:
                        format_issues.append(f"{col}: {invalid_dates} 个无效日期")
                except Exception as e:
                    format_issues.append(f"{col}: 日期格式转换失败 - {e}")
        
        # 2. 检查数值字段的格式
        numeric_columns = ['年龄', '身高', '体重', '检测孕周', '孕妇BMI']
        for col in numeric_columns:
            if col in self.cleaned_data.columns:
                # 检测孕周已经是数值类型，其他字段确保是数值类型
                if col != '检测孕周':
                    self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='coerce')
                invalid_numeric = self.cleaned_data[col].isnull().sum()
                if invalid_numeric > 0:
                    format_issues.append(f"{col}: {invalid_numeric} 个无效数值")
        
        # 3. 检查分类字段
        categorical_columns = ['IVF妊娠', '胎儿是否健康']
        for col in categorical_columns:
            if col in self.cleaned_data.columns:
                unique_values = self.cleaned_data[col].unique()
                print(f"{col} 的唯一值: {unique_values}")
        
        if format_issues:
            print("发现的格式问题:")
            for issue in format_issues:
                print(f"  - {issue}")
        else:
            print("未发现格式问题")
        
        self.cleaning_report['format_issues'] = format_issues
    
    def generate_cleaning_summary(self):
        """生成清洗总结"""
        print("\n=== 数据清洗总结 ===")
        
        original_shape = self.original_data.shape
        cleaned_shape = self.cleaned_data.shape
        
        print(f"原始数据: {original_shape[0]} 行 × {original_shape[1]} 列")
        print(f"清洗后数据: {cleaned_shape[0]} 行 × {cleaned_shape[1]} 列")
        print(f"删除记录数: {original_shape[0] - cleaned_shape[0]}")
        
        # 数据质量改善
        original_missing = self.original_data.isnull().sum().sum()
        cleaned_missing = self.cleaned_data.isnull().sum().sum()
        
        print(f"\n缺失值: {original_missing} -> {cleaned_missing}")
        print(f"数据完整性: {((cleaned_shape[0] * cleaned_shape[1] - cleaned_missing) / (cleaned_shape[0] * cleaned_shape[1]) * 100):.2f}%")
        
        return {
            'original_shape': original_shape,
            'cleaned_shape': cleaned_shape,
            'records_removed': original_shape[0] - cleaned_shape[0],
            'missing_values_before': original_missing,
            'missing_values_after': cleaned_missing,
            'data_completeness': ((cleaned_shape[0] * cleaned_shape[1] - cleaned_missing) / (cleaned_shape[0] * cleaned_shape[1]) * 100)
        }
    
    def save_cleaned_data(self, output_path='final_cleaned_data.xlsx'):
        """保存清洗后的数据"""
        if self.cleaned_data is not None:
            self.cleaned_data.to_excel(output_path, index=False)
            print(f"\n清洗后的数据已保存到: {output_path}")
            return True
        else:
            print("没有清洗后的数据可保存")
            return False
    
    def create_visualization(self):
        """创建数据清洗前后的对比可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NIPT数据清洗前后对比', fontsize=16)
        
        # 1. 缺失值对比
        original_missing = self.original_data.isnull().sum()
        cleaned_missing = self.cleaned_data.isnull().sum()
        
        missing_comparison = pd.DataFrame({
            '清洗前': original_missing,
            '清洗后': cleaned_missing
        })
        missing_comparison = missing_comparison[missing_comparison.sum(axis=1) > 0]
        
        if not missing_comparison.empty:
            missing_comparison.plot(kind='bar', ax=axes[0,0])
            axes[0,0].set_title('缺失值对比')
            axes[0,0].set_ylabel('缺失值数量')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. 数据分布对比（以年龄为例）
        axes[0,1].hist(self.original_data['年龄'], alpha=0.7, label='清洗前', bins=20)
        axes[0,1].hist(self.cleaned_data['年龄'], alpha=0.7, label='清洗后', bins=20)
        axes[0,1].set_title('年龄分布对比')
        axes[0,1].set_xlabel('年龄')
        axes[0,1].set_ylabel('频次')
        axes[0,1].legend()
        
        # 3. BMI分布对比
        axes[1,0].hist(self.original_data['孕妇BMI'], alpha=0.7, label='清洗前', bins=20)
        axes[1,0].hist(self.cleaned_data['孕妇BMI'], alpha=0.7, label='清洗后', bins=20)
        axes[1,0].set_title('BMI分布对比')
        axes[1,0].set_xlabel('BMI')
        axes[1,0].set_ylabel('频次')
        axes[1,0].legend()
        
        # 4. 数据完整性饼图
        cleaned_total = self.cleaned_data.shape[0] * self.cleaned_data.shape[1]
        cleaned_missing_total = self.cleaned_data.isnull().sum().sum()
        cleaned_complete = cleaned_total - cleaned_missing_total
        
        axes[1,1].pie([cleaned_complete, cleaned_missing_total], 
                     labels=['完整数据', '缺失数据'],
                     autopct='%1.1f%%',
                     startangle=90)
        axes[1,1].set_title('清洗后数据完整性')
        
        plt.tight_layout()
        plt.savefig('data_cleaning_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("数据清洗对比图已保存为: data_cleaning_comparison.png")

def main():
    """主函数"""
    print("NIPT数据清洗程序启动...")
    print("处理包含转换后孕周天数的数据...")
    
    # 初始化清洗器，使用已转换孕周的数据文件
    input_file = '../../cleaned_data.xlsx'
    # 如果相对路径不存在，尝试绝对路径
    import os
    if not os.path.exists(input_file):
        input_file = 'd:/Program code/pythonproject/mathmodel/cleaned_data.xlsx'
    
    cleaner = NIPTDataCleaner(input_file)
    
    # 加载数据
    if not cleaner.load_data():
        return
    
    # 分析数据质量
    cleaner.analyze_data_quality()
    
    # 执行数据清洗
    cleaner.clean_missing_values()
    cleaner.remove_duplicates()
    cleaner.validate_data_formats()
    
    # 生成清洗总结
    summary = cleaner.generate_cleaning_summary()
    
    # 保存清洗后的数据
    cleaner.save_cleaned_data()
    
    # 创建可视化
    cleaner.create_visualization()
    
    print("\n数据清洗完成！")
    
    return cleaner, summary

if __name__ == "__main__":
    cleaner, summary = main()