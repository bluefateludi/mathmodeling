#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT问题1主执行脚本
运行Y染色体浓度与孕周、BMI相关性分析
"""

import os
import sys

# 添加代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from correlation_analysis import NIPTCorrelationAnalysis

def main():
    """
    主函数
    """
    print("="*60)
    print("NIPT问题1：Y染色体浓度与孕周、BMI相关性分析")
    print("="*60)
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = "d:/Program code/pythonproject/mathmodel/final_cleaned_data.xlsx"
    output_dir = current_dir
    
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在 - {data_path}")
        return False
    
    try:
        # 创建分析实例
        analyzer = NIPTCorrelationAnalysis(data_path)
        
        # 运行完整分析
        success = analyzer.run_complete_analysis(os.path.join(output_dir, "results"))
        
        if success:
            print("\n" + "="*60)
            print("分析完成！结果文件已保存到以下位置：")
            print(f"- 分析报告：{os.path.join(output_dir, 'results', 'analysis_report.md')}")
            print(f"- 可视化图表：{os.path.join(output_dir, 'results', 'correlation_analysis_plots.png')}")
            print("="*60)
            return True
        else:
            print("分析失败！")
            return False
            
    except Exception as e:
        print(f"运行过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()