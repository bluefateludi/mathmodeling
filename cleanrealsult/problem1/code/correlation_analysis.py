#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT问题1：Y染色体浓度与孕周、BMI相关性分析
作者：数学建模团队
日期：2025年
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NIPTCorrelationAnalysis:
    """
    NIPT相关性分析类
    """
    
    def __init__(self, data_path):
        """
        初始化分析类
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.model = None
        self.results = {}
        
    def load_data(self):
        """
        加载数据
        """
        try:
            self.data = pd.read_excel(self.data_path)
            print(f"数据加载成功，共{len(self.data)}条记录")
            print(f"数据列名：{list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"数据加载失败：{e}")
            return False
    
    def explore_data(self):
        """
        数据探索性分析
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        print("\n=== 数据基本信息 ===")
        print(self.data.info())
        
        print("\n=== 数据描述性统计 ===")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numeric_cols].describe())
        
        print("\n=== 缺失值检查 ===")
        missing_data = self.data.isnull().sum()
        print(missing_data[missing_data > 0])
        
        # 保存探索结果
        self.results['data_info'] = {
            'total_records': len(self.data),
            'columns': list(self.data.columns),
            'numeric_columns': list(numeric_cols),
            'missing_values': missing_data.to_dict()
        }
    
    def correlation_analysis(self):
        """
        相关性分析
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        # 确定关键变量
        key_vars = ['Y染色体浓度', '孕周', 'BMI']
        
        # 检查变量是否存在
        available_vars = [var for var in key_vars if var in self.data.columns]
        if len(available_vars) < 3:
            print(f"关键变量不完整，可用变量：{available_vars}")
            # 尝试寻找相似的列名
            for var in key_vars:
                if var not in self.data.columns:
                    similar_cols = [col for col in self.data.columns if var.replace('染色体', '') in col or var.replace('Y', '') in col]
                    if similar_cols:
                        print(f"可能的{var}列：{similar_cols}")
        
        # 使用实际可用的数值列进行分析
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_data = self.data[numeric_cols]
        
        print("\n=== 相关性分析 ===")
        
        # 计算相关系数矩阵
        corr_matrix = correlation_data.corr()
        print("相关系数矩阵：")
        print(corr_matrix)
        
        # 如果Y染色体浓度列存在，进行具体分析
        y_col = None
        for col in self.data.columns:
            if 'Y' in col and ('染色体' in col or '浓度' in col):
                y_col = col
                break
        
        if y_col:
            print(f"\n=== {y_col}与其他变量的相关性 ===")
            correlations = corr_matrix[y_col].sort_values(ascending=False)
            print(correlations)
            
            # 计算显著性检验
            for col in numeric_cols:
                if col != y_col and not self.data[col].isna().all():
                    try:
                        corr_coef, p_value = stats.pearsonr(self.data[y_col].dropna(), 
                                                           self.data[col].dropna())
                        print(f"{y_col} vs {col}: r={corr_coef:.4f}, p={p_value:.4f}")
                    except:
                        continue
        
        # 保存相关性结果
        self.results['correlation'] = {
            'correlation_matrix': corr_matrix.to_dict(),
            'target_variable': y_col
        }
        
        return corr_matrix
    
    def build_regression_model(self):
        """
        建立多元回归模型
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        # 寻找目标变量
        y_col = None
        for col in self.data.columns:
            if 'Y' in col and ('染色体' in col or '浓度' in col):
                y_col = col
                break
        
        if not y_col:
            print("未找到Y染色体浓度相关列")
            return
        
        # 寻找特征变量
        feature_cols = []
        for col in self.data.columns:
            if '孕周' in col or 'BMI' in col or '年龄' in col:
                if self.data[col].dtype in [np.float64, np.int64]:
                    feature_cols.append(col)
        
        if len(feature_cols) == 0:
            print("未找到合适的特征变量")
            return
        
        print(f"\n=== 多元回归建模 ===")
        print(f"目标变量：{y_col}")
        print(f"特征变量：{feature_cols}")
        
        # 准备数据
        X = self.data[feature_cols].dropna()
        y = self.data.loc[X.index, y_col]
        
        # 建立模型
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # 预测
        y_pred = self.model.predict(X)
        
        # 模型评估
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"\n=== 模型评估结果 ===")
        print(f"R² 决定系数: {r2:.4f}")
        print(f"RMSE 均方根误差: {rmse:.4f}")
        print(f"回归系数: {dict(zip(feature_cols, self.model.coef_))}")
        print(f"截距: {self.model.intercept_:.4f}")
        
        # 构建回归方程
        equation = f"{y_col} = {self.model.intercept_:.4f}"
        for i, col in enumerate(feature_cols):
            coef = self.model.coef_[i]
            sign = '+' if coef >= 0 else ''
            equation += f" {sign}{coef:.4f}*{col}"
        
        print(f"\n回归方程：{equation}")
        
        # 保存模型结果
        self.results['regression'] = {
            'target_variable': y_col,
            'feature_variables': feature_cols,
            'r2_score': r2,
            'rmse': rmse,
            'coefficients': dict(zip(feature_cols, self.model.coef_)),
            'intercept': self.model.intercept_,
            'equation': equation,
            'sample_size': len(X)
        }
        
        return self.model
    
    def create_correlation_heatmap(self, save_path=None):
        """
        创建相关性热力图（单独显示，优化版本）
        
        Args:
            save_path (str): 图表保存路径
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        # 创建单独的相关性热力图
        plt.figure(figsize=(14, 12))
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            
            # 创建优化的热力图，移除数字标注，优化图例
            ax = sns.heatmap(corr_matrix, 
                           annot=False,  # 移除数字标注
                           cmap='RdBu_r',  # 使用更清晰的颜色映射
                           center=0, 
                           square=True, 
                           linewidths=1.0,  # 增加网格线宽度
                           linecolor='white',  # 设置网格线颜色
                           cbar_kws={
                               "shrink": 0.8, 
                               "aspect": 30,
                               "label": "相关系数",
                               "orientation": "vertical"
                           })
            
            # 优化标题和标签
            plt.title('NIPT问题1：变量相关性热力图', 
                     fontsize=18, fontweight='bold', pad=25)
            
            # 优化坐标轴标签
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
            
            # 添加图例说明
            plt.figtext(0.02, 0.02, 
                       '图例说明：红色表示正相关，蓝色表示负相关，颜色深度表示相关性强度',
                       fontsize=10, style='italic', wrap=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"优化后的相关性热力图已保存至：{save_path}")
        
        plt.show()
    
    def create_analysis_plots(self, save_path=None):
        """
        创建分析图表（Y染色体分布图和散点图组合）
        
        Args:
            save_path (str): 图表保存路径
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        # 创建1行3列的子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('NIPT问题1：Y染色体浓度分析', fontsize=16, fontweight='bold')
        
        # 获取Y染色体相关列
        y_col = None
        for col in self.data.columns:
            if 'Y' in col and ('染色体' in col or '浓度' in col or 'Z值' in col):
                y_col = col
                break
        
        # 1. Y染色体浓度分布
        if y_col:
            self.data[y_col].hist(bins=30, ax=axes[0], alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_title(f'{y_col}分布', fontsize=14, fontweight='bold')
            axes[0].set_xlabel(y_col, fontsize=12)
            axes[0].set_ylabel('频数', fontsize=12)
            axes[0].grid(True, alpha=0.3)
        
        # 2. 散点图：Y染色体浓度 vs 孕周
        week_col = None
        for col in self.data.columns:
            if '孕周' in col:
                week_col = col
                break
        
        if y_col and week_col:
            axes[1].scatter(self.data[week_col], self.data[y_col], alpha=0.6, color='blue', s=20)
            axes[1].set_xlabel(week_col, fontsize=12)
            axes[1].set_ylabel(y_col, fontsize=12)
            axes[1].set_title(f'{y_col} vs {week_col}', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # 添加趋势线
            if not self.data[week_col].isna().all() and not self.data[y_col].isna().all():
                valid_data = self.data[[week_col, y_col]].dropna()
                if len(valid_data) > 1:
                    x_data = valid_data[week_col]
                    y_data = valid_data[y_col]
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    axes[1].plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
        
        # 3. 散点图：Y染色体浓度 vs BMI
        bmi_col = None
        for col in self.data.columns:
            if 'BMI' in col:
                bmi_col = col
                break
        
        if y_col and bmi_col:
            axes[2].scatter(self.data[bmi_col], self.data[y_col], alpha=0.6, color='green', s=20)
            axes[2].set_xlabel(bmi_col, fontsize=12)
            axes[2].set_ylabel(y_col, fontsize=12)
            axes[2].set_title(f'{y_col} vs {bmi_col}', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            # 添加趋势线
            if not self.data[bmi_col].isna().all() and not self.data[y_col].isna().all():
                valid_data = self.data[[bmi_col, y_col]].dropna()
                if len(valid_data) > 1:
                    x_data = valid_data[bmi_col]
                    y_data = valid_data[y_col]
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    axes[2].plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分析图表已保存至：{save_path}")
        
        plt.show()
    
    def create_visualizations(self, save_path=None):
        """
        创建所有可视化图表（保持向后兼容性）
        
        Args:
            save_path (str): 图表保存路径
        """
        if save_path:
            # 分别保存两个图表
            base_path = save_path.rsplit('.', 1)[0]
            ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
            
            heatmap_path = f"{base_path}_heatmap.{ext}"
            analysis_path = f"{base_path}_analysis.{ext}"
            
            self.create_correlation_heatmap(heatmap_path)
            self.create_analysis_plots(analysis_path)
        else:
            self.create_correlation_heatmap()
            self.create_analysis_plots()
    
    def generate_report(self, save_path=None):
        """
        生成分析报告
        
        Args:
            save_path (str): 报告保存路径
        """
        if not self.results:
            print("请先完成数据分析")
            return
        
        report = []
        report.append("# NIPT问题1：Y染色体浓度相关性分析报告\n")
        report.append("## 1. 数据概况\n")
        
        if 'data_info' in self.results:
            info = self.results['data_info']
            report.append(f"- 总记录数：{info['total_records']}")
            report.append(f"- 数据列数：{len(info['columns'])}")
            report.append(f"- 数值型变量：{len(info['numeric_columns'])}个")
            report.append("\n")
        
        report.append("## 2. 相关性分析结果\n")
        
        if 'correlation' in self.results:
            target = self.results['correlation']['target_variable']
            if target:
                report.append(f"- 目标变量：{target}")
                report.append("- 相关性分析显示了各变量间的线性关系强度")
                report.append("\n")
        
        report.append("## 3. 回归模型结果\n")
        
        if 'regression' in self.results:
            reg = self.results['regression']
            report.append(f"- **回归方程**：{reg['equation']}")
            report.append(f"- **R²决定系数**：{reg['r2_score']:.4f}")
            report.append(f"- **RMSE**：{reg['rmse']:.4f}")
            report.append(f"- **样本量**：{reg['sample_size']}")
            report.append("\n### 回归系数解释：")
            
            for var, coef in reg['coefficients'].items():
                direction = "正相关" if coef > 0 else "负相关"
                report.append(f"- {var}：系数={coef:.4f}，与目标变量呈{direction}")
            report.append("\n")
        
        report.append("## 4. 结论与建议\n")
        report.append("- 通过相关性分析和回归建模，我们建立了Y染色体浓度与孕周、BMI等因素的定量关系")
        report.append("- 模型可用于预测和理解影响Y染色体浓度的关键因素")
        report.append("- 建议在实际应用中结合临床经验进行综合判断")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存至：{save_path}")
        
        return report_text
    
    def run_complete_analysis(self, output_dir=None):
        """
        运行完整分析流程
        
        Args:
            output_dir (str): 输出目录
        """
        print("开始NIPT问题1完整分析...")
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 数据探索
        self.explore_data()
        
        # 3. 相关性分析
        self.correlation_analysis()
        
        # 4. 回归建模
        self.build_regression_model()
        
        # 5. 生成可视化
        if output_dir:
            viz_path = f"{output_dir}/correlation_analysis_plots.png"
            self.create_visualizations(viz_path)
        else:
            self.create_visualizations()
        
        # 6. 生成报告
        if output_dir:
            report_path = f"{output_dir}/analysis_report.md"
            self.generate_report(report_path)
        else:
            print("\n" + self.generate_report())
        
        print("\n分析完成！")
        return True


if __name__ == "__main__":
    # 使用示例
    data_path = "d:/Program code/pythonproject/mathmodel/final_cleaned_data.xlsx"
    output_dir = "d:/Program code/pythonproject/mathmodel/cleanrealsult/problem1"
    
    # 创建分析实例
    analyzer = NIPTCorrelationAnalysis(data_path)
    
    # 运行完整分析
    analyzer.run_complete_analysis(output_dir)