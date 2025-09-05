#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测孕周转换为小数周数处理脚本

基于gestational_week_fix.py，将检测孕周字段转换为小数周数
并保持其他清理后的数据不变。

作者: AI助手
日期: 2025年1月
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GestationalWeekToDecimalProcessor:
    """
    检测孕周转小数周数处理器
    
    用于将NIPT数据中的检测孕周字段转换为小数周数
    """
    
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        self.conversion_log = []
    
    def parse_gestational_week_to_decimal(self, week_str):
        """
        解析孕周字符串，转换为小数周数
        
        参数:
        week_str: 孕周字符串，如 '11w+6', '13w', '20w+1'
        
        返回:
        float: 孕周对应的小数周数
        """
        if pd.isna(week_str) or week_str == '':
            return np.nan
        
        week_str = str(week_str).strip().lower()
        original_str = week_str
        
        try:
            # 处理 'w+d' 格式（如 '11w+6'）
            if 'w+' in week_str:
                parts = week_str.split('w+')
                if len(parts) == 2:
                    weeks = int(parts[0])
                    days = int(parts[1])
                    decimal_weeks = round(weeks + days / 7.0, 2)
                    self.conversion_log.append({
                        'original': original_str,
                        'parsed_weeks': decimal_weeks,
                        'format': 'w+d',
                        'weeks': weeks,
                        'days': days
                    })
                    self.processed_count += 1
                    return decimal_weeks
            
            # 处理 'w' 格式（如 '13w'）
            elif 'w' in week_str and '+' not in week_str:
                weeks_str = week_str.replace('w', '')
                weeks = int(weeks_str)
                decimal_weeks = float(weeks)
                self.conversion_log.append({
                    'original': original_str,
                    'parsed_weeks': decimal_weeks,
                    'format': 'w',
                    'weeks': weeks,
                    'days': 0
                })
                self.processed_count += 1
                return decimal_weeks
            
            # 处理纯数字格式（假设是周数）
            else:
                weeks = float(week_str)
                decimal_weeks = round(weeks, 2)
                self.conversion_log.append({
                    'original': original_str,
                    'parsed_weeks': decimal_weeks,
                    'format': 'numeric',
                    'weeks': int(weeks),
                    'days': int((weeks - int(weeks)) * 7)
                })
                self.processed_count += 1
                return decimal_weeks
                
        except (ValueError, IndexError) as e:
            self.error_count += 1
            print(f"解析错误: '{original_str}' - {str(e)}")
            return np.nan
    
    def process_dataframe(self, df, column_name='检测孕周'):
        """
        处理DataFrame中的检测孕周列，转换为小数周数
        
        参数:
        df: pandas DataFrame
        column_name: 检测孕周列名
        
        返回:
        pandas DataFrame: 处理后的数据框
        """
        if column_name not in df.columns:
            raise ValueError(f"列 '{column_name}' 不存在于数据框中")
        
        print(f"开始处理 {column_name} 字段，转换为小数周数...")
        print(f"原始数据量: {len(df)} 行")
        print(f"原始 {column_name} 非空值: {df[column_name].notna().sum()} 个")
        
        # 创建新列保存转换结果（小数周数）
        df[f'{column_name}_小数周数'] = df[column_name].apply(self.parse_gestational_week_to_decimal)
        
        # 数据验证
        parsed_data = df[f'{column_name}_小数周数'].dropna()
        if len(parsed_data) > 0:
            min_days = parsed_data.min()
            max_days = parsed_data.max()
            mean_days = parsed_data.mean()
            
            print(f"\n解析结果统计:")
            print(f"成功解析: {self.processed_count} 个")
            print(f"解析失败: {self.error_count} 个")
            print(f"解析成功率: {self.processed_count/(self.processed_count+self.error_count)*100:.2f}%")
            print(f"小数周数范围: {min_days:.2f} - {max_days:.2f} 周")
            print(f"平均小数周数: {mean_days:.2f} 周")
            
            # 检查异常值（正常孕周应该在10-42周之间）
            abnormal_weeks = parsed_data[(parsed_data < 10) | (parsed_data > 42)]
            if len(abnormal_weeks) > 0:
                print(f"\n警告: 发现 {len(abnormal_weeks)} 个异常小数周数值:")
                print(abnormal_weeks.tolist())
        
        return df

def process_gestational_week_to_decimal():
    """
    主处理函数：将检测孕周转换为小数周数并保存结果
    """
    print("=== 检测孕周转小数周数处理程序 ===")
    print("正在将检测孕周字段转换为小数周数...\n")
    
    # 文件路径
    base_path = Path("../../")
    input_file = base_path / "CUMCM2025Problems/C题/附件.xlsx"
    output_file = base_path / "CUMCM2025Problems/C题/Women.xlsx"
    
    # 如果相对路径不存在，尝试绝对路径
    if not input_file.exists():
        base_path = Path("d:/Program code/pythonproject/mathmodel")
        input_file = base_path / "CUMCM2025Problems/C题/附件.xlsx"
        output_file = base_path / "CUMCM2025Problems/C题/Women.xlsx"
    
    try:
        # 1. 加载原始数据
        print("1. 加载原始数据...")
        df = pd.read_excel(input_file)
        print(f"   原始数据形状: {df.shape}")
        print(f"   列名: {list(df.columns)}")
        
        # 2. 创建处理器并处理数据
        print("\n2. 处理检测孕周字段...")
        processor = GestationalWeekToDecimalProcessor()
        df_processed = processor.process_dataframe(df.copy())
        
        # 3. 替换原始检测孕周列为小数周数列
        if '检测孕周_小数周数' in df_processed.columns:
            # 将小数周数列重命名为原列名，覆盖原始数据
            df_processed['检测孕周'] = df_processed['检测孕周_小数周数']
            # 删除临时列
            df_processed = df_processed.drop('检测孕周_小数周数', axis=1)
            print("\n3. 已将检测孕周字段替换为小数周数")
        
        # 4. 数据验证
        print("\n4. 数据验证...")
        gestational_weeks = df_processed['检测孕周'].dropna()
        
        if len(gestational_weeks) > 0:
            # 检查合理性（正常孕周10-42周）
            reasonable_weeks = gestational_weeks[(gestational_weeks >= 10) & (gestational_weeks <= 42)]
            reasonable_rate = len(reasonable_weeks) / len(gestational_weeks) * 100
            
            print(f"   转换成功的小数周数数据: {len(gestational_weeks)} 个")
            print(f"   合理范围内的数据: {len(reasonable_weeks)} 个 ({reasonable_rate:.2f}%)")
            print(f"   小数周数统计: 最小={gestational_weeks.min():.2f}, 最大={gestational_weeks.max():.2f}, 平均={gestational_weeks.mean():.2f}")
        
        # 5. 保存处理后的数据
        print("\n5. 保存处理后的数据...")
        df_processed.to_excel(output_file, index=False)
        print(f"   处理后数据已保存到: {output_file}")
        
        # 6. 显示转换示例
        print("\n6. 转换示例:")
        if processor.conversion_log:
            for i, log in enumerate(processor.conversion_log[:5]):  # 显示前5个示例
                print(f"   '{log['original']}' → {log['parsed_weeks']} 周")
            if len(processor.conversion_log) > 5:
                print(f"   ... 还有 {len(processor.conversion_log) - 5} 个转换记录")
        
        print("\n=== 处理完成 ===")
        print("检测孕周字段已成功转换为小数周数并保存到Women.xlsx")
        print("\n主要改进:")
        print(f"- 原始有效数据: {df['检测孕周'].notna().sum()} 个")
        print(f"- 转换后有效数据: {df_processed['检测孕周'].notna().sum()} 个")
        print(f"- 数据保留率: {df_processed['检测孕周'].notna().sum()/df['检测孕周'].notna().sum()*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请检查文件路径和数据格式是否正确。")
        return False

if __name__ == "__main__":
    success = process_gestational_week_to_decimal()
    if success:
        print("\n程序执行成功！")
    else:
        print("\n程序执行失败，请检查错误信息。")