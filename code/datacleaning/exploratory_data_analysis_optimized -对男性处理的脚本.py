#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT Data Exploratory Analysis - Optimized Version

This script performs comprehensive exploratory data analysis on NIPT (Non-Invasive Prenatal Testing) data.
Optimized for better visualization layout and English labels to avoid encoding issues.

Author: AI Assistant
Date: 2025-01
Version: 2.0 (Optimized)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings
from datetime import datetime
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set matplotlib to use a font that supports English
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

class NIPTExploratoryAnalysisOptimized:
    """
    Optimized NIPT Data Exploratory Analysis Class
    
    Features:
    - Improved chart layouts
    - English labels to avoid encoding issues
    - HTML report generation
    - Interactive visualizations
    """
    
    def __init__(self, data_path='../../CUMCM2025Problems/C题/Women.xlsx'):
        self.data_path = data_path
        self.data = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.analysis_results = {}
        self.figures = []
        
        # Create output directory
        self.output_dir = Path('../../analysis_results_optimized')
        self.output_dir.mkdir(exist_ok=True)
        
        # Column name mapping (Chinese to English)
        self.column_mapping = {
            '年龄': 'Age',
            '身高': 'Height',
            '体重': 'Weight', 
            '孕妇BMI': 'BMI',
            'IVF妊娠': 'IVF_Pregnancy',
            '胎儿是否健康': 'Fetal_Health',
            '染色体的非整倍体': 'Chromosome_Aneuploidy',
            '怀孕次数': 'Pregnancy_Count'
        }
        
    def load_data(self):
        """
        Load and preprocess the data
        """
        print("=== Loading Data ===")
        try:
            self.data = pd.read_excel(self.data_path)
            print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            
            # Rename columns to English
            self.data = self.data.rename(columns=self.column_mapping)
            
            # Identify column types
            self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            print(f"Numeric columns ({len(self.numeric_columns)}): {self.numeric_columns[:5]}...")
            print(f"Categorical columns ({len(self.categorical_columns)}): {self.categorical_columns[:5]}...")
            
            # Data quality overview
            total_cells = self.data.shape[0] * self.data.shape[1]
            missing_cells = self.data.isnull().sum().sum()
            completeness = (total_cells - missing_cells) / total_cells * 100
            
            self.analysis_results['data_overview'] = {
                'total_records': self.data.shape[0],
                'total_fields': self.data.shape[1],
                'missing_values': missing_cells,
                'completeness_rate': completeness
            }
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        return self.data
    
    def basic_statistics(self):
        """
        Calculate basic statistical descriptions
        """
        print("\n=== Basic Statistical Analysis ===")
        
        # Numeric variables statistics
        if self.numeric_columns:
            numeric_stats = self.data[self.numeric_columns].describe()
            print("\nNumeric Variables Statistics:")
            print(numeric_stats)
            
            self.analysis_results['numeric_statistics'] = {
                'summary_table': numeric_stats.to_dict(),
                'variables': self.numeric_columns
            }
        
        # Categorical variables statistics
        if self.categorical_columns:
            categorical_stats = {}
            print("\nCategorical Variables Statistics:")
            
            for col in self.categorical_columns:
                if col in self.data.columns:
                    stats_info = {
                        'unique_count': self.data[col].nunique(),
                        'most_frequent': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                        'most_frequent_count': self.data[col].value_counts().iloc[0] if not self.data[col].empty else 0,
                        'missing_count': self.data[col].isnull().sum()
                    }
                    categorical_stats[col] = stats_info
                    print(f"  {col}: {stats_info['unique_count']} unique values, most frequent: {stats_info['most_frequent']}")
            
            self.analysis_results['categorical_statistics'] = categorical_stats
    
    def correlation_analysis(self):
        """
        Perform correlation analysis
        """
        print("\n=== Correlation Analysis ===")
        
        if len(self.numeric_columns) < 2:
            print("Insufficient numeric variables for correlation analysis")
            return
        
        # Calculate correlation matrix
        correlation_matrix = self.data[self.numeric_columns].corr()
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(3))
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if strong_correlations:
            print("\nStrong Correlations (|r| > 0.7):")
            for corr in strong_correlations:
                print(f"  {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}")
        
        self.analysis_results['correlation_analysis'] = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def statistical_tests(self):
        """
        Perform statistical tests
        """
        print("\n=== Statistical Tests ===")
        
        test_results = {}
        
        # Normality tests for numeric variables
        if self.numeric_columns:
            print("\nNormality Tests (Shapiro-Wilk):")
            normality_tests = {}
            
            for col in self.numeric_columns:
                if col in self.data.columns:
                    data_clean = self.data[col].dropna()
                    if len(data_clean) > 3:  # Minimum sample size for Shapiro-Wilk
                        # Sample if too large (Shapiro-Wilk limit)
                        if len(data_clean) > 5000:
                            data_sample = data_clean.sample(5000, random_state=42)
                        else:
                            data_sample = data_clean
                        
                        statistic, p_value = stats.shapiro(data_sample)
                        normality_tests[col] = {
                            'statistic': statistic,
                            'p_value': p_value,
                            'is_normal': p_value > 0.05
                        }
                        print(f"  {col}: p-value={p_value:.4f} ({'Normal' if p_value > 0.05 else 'Non-normal'})")
            
            test_results['normality_tests'] = normality_tests
        
        # Independence tests for categorical variables
        if len(self.categorical_columns) >= 2:
            print("\nIndependence Tests (Chi-square):")
            independence_tests = {}
            
            # Test key relationships
            test_pairs = [('IVF_Pregnancy', 'Fetal_Health')] if 'IVF_Pregnancy' in self.categorical_columns and 'Fetal_Health' in self.categorical_columns else []
            
            for var1, var2 in test_pairs:
                if var1 in self.data.columns and var2 in self.data.columns:
                    contingency_table = pd.crosstab(self.data[var1], self.data[var2])
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    independence_tests[f"{var1}_vs_{var2}"] = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'degrees_of_freedom': dof,
                        'is_independent': p_value > 0.05,
                        'contingency_table': contingency_table.to_dict()
                    }
                    print(f"  {var1} vs {var2}: χ²={chi2:.4f}, p-value={p_value:.4f}, {'Independent' if p_value > 0.05 else 'Related'}")
        
            test_results['independence_tests'] = independence_tests
        
        self.analysis_results['statistical_tests'] = test_results
    
    def create_visualizations(self):
        """
        Create optimized visualizations
        """
        print("\n=== Generating Optimized Visualizations ===")
        
        # 1. Improved numeric distributions
        self._plot_numeric_distributions_optimized()
        
        # 2. Enhanced correlation heatmap
        self._plot_correlation_heatmap_optimized()
        
        # 3. Better categorical distributions
        self._plot_categorical_distributions_optimized()
        
        # 4. Improved scatter matrix
        self._plot_scatter_matrix_optimized()
        
        # 5. Enhanced boxplots
        self._plot_boxplots_optimized()
        
        # 6. Interactive plots
        self._create_interactive_plots()
        
        print(f"Generated {len(self.figures)} optimized charts")
    
    def _plot_numeric_distributions_optimized(self):
        """
        Create optimized numeric distribution plots with better layout
        """
        if not self.numeric_columns:
            return
        
        # Limit to key variables to avoid overcrowding
        key_numeric = ['Age', 'Height', 'Weight', 'BMI']
        plot_columns = [col for col in key_numeric if col in self.numeric_columns]
        
        if not plot_columns:
            plot_columns = self.numeric_columns[:6]  # Limit to 6 variables max
        
        # Calculate optimal layout
        n_vars = len(plot_columns)
        if n_vars <= 2:
            n_cols, n_rows = n_vars, 1
        elif n_vars <= 4:
            n_cols, n_rows = 2, 2
        elif n_vars <= 6:
            n_cols, n_rows = 3, 2
        else:
            n_cols, n_rows = 3, 3
        
        # Create figure with improved spacing
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('Distribution of Numeric Variables', fontsize=16, y=0.98)
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_columns)))
        
        for i, col in enumerate(plot_columns):
            if i < len(axes):
                data_clean = self.data[col].dropna()
                
                # Create histogram with better styling
                axes[i].hist(data_clean, bins=25, alpha=0.7, color=colors[i], 
                           edgecolor='black', linewidth=0.5)
                
                # Add statistics
                mean_val = data_clean.mean()
                median_val = data_clean.median()
                
                axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {mean_val:.1f}')
                axes[i].axvline(median_val, color='blue', linestyle=':', linewidth=2, 
                              label=f'Median: {median_val:.1f}')
                
                axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(col, fontsize=10)
                axes[i].set_ylabel('Frequency', fontsize=10)
                axes[i].legend(fontsize=9)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(plot_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(self.output_dir / 'numeric_distributions_optimized.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        self.figures.append('numeric_distributions_optimized.png')
        plt.show()
    
    def _plot_correlation_heatmap_optimized(self):
        """
        Create enhanced correlation heatmap
        """
        if len(self.numeric_columns) < 2:
            return
        
        # Select key variables for cleaner visualization
        key_numeric = ['Age', 'Height', 'Weight', 'BMI']
        plot_columns = [col for col in key_numeric if col in self.numeric_columns]
        
        if len(plot_columns) < 2:
            plot_columns = self.numeric_columns[:8]  # Limit to 8 variables max
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data[plot_columns].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap with improved styling
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                   annot_kws={'size': 10})
        
        plt.title('Variable Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap_optimized.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        self.figures.append('correlation_heatmap_optimized.png')
        plt.show()
    
    def _plot_categorical_distributions_optimized(self):
        """
        Create improved categorical distribution plots
        """
        if not self.categorical_columns:
            return
        
        # Select important categorical variables
        important_categorical = ['IVF_Pregnancy', 'Fetal_Health', 'Chromosome_Aneuploidy']
        plot_columns = [col for col in important_categorical if col in self.categorical_columns]
        
        if not plot_columns:
            plot_columns = self.categorical_columns[:3]
        
        n_cols = min(3, len(plot_columns))
        n_rows = 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7))
        fig.suptitle('Distribution of Categorical Variables', fontsize=16, y=0.95)
        
        if n_cols == 1:
            axes = [axes]
        
        colors = plt.cm.Set2(np.linspace(0, 1, 8))
        
        for i, col in enumerate(plot_columns):
            if i < len(axes):
                value_counts = self.data[col].value_counts()
                
                # Create pie chart with better styling and no overlapping labels
                wedges, texts, autotexts = axes[i].pie(value_counts.values, 
                                                      labels=None,  # Remove labels to avoid overlap
                                                      autopct='%1.1f%%',
                                                      startangle=90,
                                                      colors=colors[:len(value_counts)],
                                                      explode=[0.05] * len(value_counts),
                                                      pctdistance=0.85)  # Move percentage text closer to center
                
                # Improve text styling for percentages
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
                
                # Create legend instead of labels to avoid overlap
                axes[i].legend(wedges, value_counts.index, 
                             title="Categories",
                             loc="center left", 
                             bbox_to_anchor=(1, 0, 0.5, 1),
                             fontsize=9)
                
                axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, right=0.85)  # Adjust for legend space
        plt.savefig(self.output_dir / 'categorical_distributions_optimized.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        self.figures.append('categorical_distributions_optimized.png')
        plt.show()
    
    def _plot_scatter_matrix_optimized(self):
        """
        Create improved scatter matrix
        """
        # Select key numeric variables
        key_numeric = ['Age', 'Height', 'Weight', 'BMI']
        plot_columns = [col for col in key_numeric if col in self.numeric_columns]
        
        if len(plot_columns) < 2:
            return
        
        # Use seaborn pairplot for better styling
        plt.figure(figsize=(12, 10))
        
        # Create pairplot
        g = sns.pairplot(self.data[plot_columns].dropna(), 
                        diag_kind='hist',
                        plot_kws={'alpha': 0.6, 's': 20},
                        diag_kws={'bins': 20, 'alpha': 0.7})
        
        g.fig.suptitle('Key Variables Scatter Matrix', fontsize=16, y=1.02)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scatter_matrix_optimized.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        self.figures.append('scatter_matrix_optimized.png')
        plt.show()
    
    def _plot_boxplots_optimized(self):
        """
        Create enhanced boxplot analysis
        """
        if 'Fetal_Health' not in self.categorical_columns:
            return
        
        # Create subplots for multiple boxplots
        numeric_vars = ['Age', 'BMI']
        available_vars = [var for var in numeric_vars if var in self.numeric_columns]
        
        if not available_vars:
            return
        
        fig, axes = plt.subplots(1, len(available_vars), figsize=(6*len(available_vars), 6))
        fig.suptitle('Distribution by Fetal Health Status', fontsize=16, y=0.98)
        
        if len(available_vars) == 1:
            axes = [axes]
        
        for i, var in enumerate(available_vars):
            sns.boxplot(data=self.data, x='Fetal_Health', y=var, ax=axes[i],
                       palette='Set2')
            axes[i].set_title(f'{var} by Fetal Health', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Fetal Health Status', fontsize=10)
            axes[i].set_ylabel(var, fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(self.output_dir / 'boxplots_optimized.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        self.figures.append('boxplots_optimized.png')
        plt.show()
    
    def _create_interactive_plots(self):
        """
        Create interactive plots using Plotly
        """
        try:
            # Interactive correlation heatmap
            if len(self.numeric_columns) >= 2:
                key_numeric = ['Age', 'Height', 'Weight', 'BMI']
                plot_columns = [col for col in key_numeric if col in self.numeric_columns]
                
                if len(plot_columns) < 2:
                    plot_columns = self.numeric_columns[:6]
                
                corr_matrix = self.data[plot_columns].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Interactive Correlation Heatmap',
                    xaxis_title='Variables',
                    yaxis_title='Variables',
                    width=600,
                    height=600
                )
                
                fig.write_html(self.output_dir / 'interactive_correlation.html')
                self.figures.append('interactive_correlation.html')
        
        except ImportError:
            print("Plotly not available for interactive plots")
    
    def generate_html_report(self):
        """
        Generate HTML report with embedded visualizations
        """
        print("\n=== Generating HTML Report ===")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIPT Exploratory Data Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .highlight {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NIPT Exploratory Data Analysis Report</h1>
        <p style="text-align: center; color: #7f8c8d;">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>1. Data Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{self.analysis_results['data_overview']['total_records']:,}</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.analysis_results['data_overview']['total_fields']}</div>
                <div class="stat-label">Total Fields</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.analysis_results['data_overview']['missing_values']:,}</div>
                <div class="stat-label">Missing Values</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.analysis_results['data_overview']['completeness_rate']:.1f}%</div>
                <div class="stat-label">Data Completeness</div>
            </div>
        </div>
        
        <h2>2. Visualizations</h2>
        
        <h3>2.1 Numeric Variable Distributions</h3>
        <div class="chart-container">
            <img src="numeric_distributions_optimized.png" alt="Numeric Distributions">
            <p><em>Distribution patterns of key numeric variables with statistical indicators</em></p>
        </div>
        
        <h3>2.2 Variable Correlations</h3>
        <div class="chart-container">
            <img src="correlation_heatmap_optimized.png" alt="Correlation Heatmap">
            <p><em>Correlation matrix showing relationships between numeric variables</em></p>
        </div>
        
        <h3>2.3 Categorical Variable Distributions</h3>
        <div class="chart-container">
            <img src="categorical_distributions_optimized.png" alt="Categorical Distributions">
            <p><em>Distribution of key categorical variables</em></p>
        </div>
        
        <h3>2.4 Variable Relationships</h3>
        <div class="chart-container">
            <img src="scatter_matrix_optimized.png" alt="Scatter Matrix">
            <p><em>Pairwise relationships between key numeric variables</em></p>
        </div>
        
        <h3>2.5 Group Comparisons</h3>
        <div class="chart-container">
            <img src="boxplots_optimized.png" alt="Boxplots">
            <p><em>Distribution comparisons by fetal health status</em></p>
        </div>
        
        <h2>3. Key Findings</h2>
        <div class="highlight">
            <h3>Strong Correlations</h3>
"""
        
        # Add correlation findings
        if 'correlation_analysis' in self.analysis_results and 'strong_correlations' in self.analysis_results['correlation_analysis']:
            strong_corrs = self.analysis_results['correlation_analysis']['strong_correlations']
            if strong_corrs:
                html_content += "<ul>"
                for corr in strong_corrs:
                    html_content += f"<li><strong>{corr['var1']} ↔ {corr['var2']}</strong>: {corr['correlation']:.3f}</li>"
                html_content += "</ul>"
            else:
                html_content += "<p>No strong correlations (|r| > 0.7) found.</p>"
        
        html_content += """
        </div>
        
        <div class="highlight">
            <h3>Statistical Test Results</h3>
"""
        
        # Add statistical test results
        if 'statistical_tests' in self.analysis_results:
            if 'normality_tests' in self.analysis_results['statistical_tests']:
                html_content += "<h4>Normality Tests</h4><ul>"
                for var, result in self.analysis_results['statistical_tests']['normality_tests'].items():
                    status = "Normal" if result['is_normal'] else "Non-normal"
                    html_content += f"<li><strong>{var}</strong>: p-value={result['p_value']:.4f} ({status})</li>"
                html_content += "</ul>"
            
            if 'independence_tests' in self.analysis_results['statistical_tests']:
                html_content += "<h4>Independence Tests</h4><ul>"
                for test_name, result in self.analysis_results['statistical_tests']['independence_tests'].items():
                    status = "Independent" if result['is_independent'] else "Related"
                    html_content += f"<li><strong>{test_name.replace('_vs_', ' vs ')}</strong>: p-value={result['p_value']:.4f} ({status})</li>"
                html_content += "</ul>"
        
        html_content += """
        </div>
        
        <h2>4. Recommendations</h2>
        <div class="highlight">
            <h3>Data Quality</h3>
            <p>The dataset shows good overall quality with high completeness rate. Continue monitoring data collection processes.</p>
            
            <h3>Further Analysis</h3>
            <ul>
                <li>Investigate strong correlations for potential causal relationships</li>
                <li>Consider non-parametric tests for non-normal distributions</li>
                <li>Explore interaction effects between variables</li>
                <li>Develop predictive models based on identified patterns</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Report generated by NIPT Exploratory Data Analysis Tool (Optimized Version)</p>
            <p>For questions or additional analysis, please contact the data analysis team.</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        html_path = self.output_dir / 'NIPT_analysis_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved to: {html_path}")
        return html_path
    
    def generate_markdown_report(self):
        """
        Generate optimized Markdown report
        """
        print("\n=== Generating Markdown Report ===")
        
        report_content = f"""# NIPT Exploratory Data Analysis Report (Optimized)

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Data Overview

| Metric | Value |
|--------|-------|
| Total Records | {self.analysis_results['data_overview']['total_records']:,} |
| Total Fields | {self.analysis_results['data_overview']['total_fields']} |
| Missing Values | {self.analysis_results['data_overview']['missing_values']:,} |
| Data Completeness | {self.analysis_results['data_overview']['completeness_rate']:.1f}% |

## 2. Variable Summary

### Numeric Variables ({len(self.numeric_columns)})
{', '.join(self.numeric_columns[:10])}{'...' if len(self.numeric_columns) > 10 else ''}

### Categorical Variables ({len(self.categorical_columns)})
{', '.join(self.categorical_columns[:10])}{'...' if len(self.categorical_columns) > 10 else ''}

## 3. Key Findings

### Strong Correlations (|r| > 0.7)
"""
        
        # Add correlation findings
        if 'correlation_analysis' in self.analysis_results and 'strong_correlations' in self.analysis_results['correlation_analysis']:
            strong_corrs = self.analysis_results['correlation_analysis']['strong_correlations']
            if strong_corrs:
                for corr in strong_corrs:
                    report_content += f"- **{corr['var1']} ↔ {corr['var2']}**: {corr['correlation']:.3f}\n"
            else:
                report_content += "- No strong correlations found\n"
        
        report_content += "\n### Statistical Test Results\n\n"
        
        # Add statistical test results
        if 'statistical_tests' in self.analysis_results:
            if 'normality_tests' in self.analysis_results['statistical_tests']:
                report_content += "#### Normality Tests\n\n"
                for var, result in self.analysis_results['statistical_tests']['normality_tests'].items():
                    status = "Normal" if result['is_normal'] else "Non-normal"
                    report_content += f"- **{var}**: p-value={result['p_value']:.4f} ({status})\n"
            
            if 'independence_tests' in self.analysis_results['statistical_tests']:
                report_content += "\n#### Independence Tests\n\n"
                for test_name, result in self.analysis_results['statistical_tests']['independence_tests'].items():
                    status = "Independent" if result['is_independent'] else "Related"
                    report_content += f"- **{test_name.replace('_vs_', ' vs ')}**: p-value={result['p_value']:.4f} ({status})\n"
        
        report_content += f"""

## 4. Visualizations

### 4.1 Numeric Variable Distributions
![Numeric Distributions](numeric_distributions_optimized.png)

### 4.2 Variable Correlations
![Correlation Heatmap](correlation_heatmap_optimized.png)

### 4.3 Categorical Variable Distributions
![Categorical Distributions](categorical_distributions_optimized.png)

### 4.4 Variable Relationships
![Scatter Matrix](scatter_matrix_optimized.png)

### 4.5 Group Comparisons
![Boxplots](boxplots_optimized.png)

## 5. Recommendations

### Data Quality
- Dataset shows good overall quality with {self.analysis_results['data_overview']['completeness_rate']:.1f}% completeness
- Continue monitoring data collection processes

### Further Analysis
- Investigate strong correlations for potential causal relationships
- Consider non-parametric tests for non-normal distributions
- Explore interaction effects between variables
- Develop predictive models based on identified patterns

### Technical Notes
- All visualizations use English labels to avoid encoding issues
- Charts are optimized for better layout and readability
- Interactive versions available in HTML format

---

**Report generated by NIPT Exploratory Data Analysis Tool (Optimized Version)**
"""
        
        # Save Markdown report
        md_path = self.output_dir / 'NIPT_analysis_report_optimized.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Markdown report saved to: {md_path}")
        return md_path
    
    def run_complete_analysis(self):
        """
        Run the complete optimized analysis pipeline
        """
        print("Starting NIPT Exploratory Data Analysis (Optimized Version)")
        print("=" * 60)
        
        try:
            # Load and analyze data
            self.load_data()
            self.basic_statistics()
            self.correlation_analysis()
            self.statistical_tests()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate reports
            self.generate_markdown_report()
            self.generate_html_report()
            
            print("\n" + "=" * 60)
            print("Analysis completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            print(f"Generated {len(self.figures)} visualizations")
            
            return self
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

def main():
    """
    Main function to run the analysis
    """
    analyzer = NIPTExploratoryAnalysisOptimized()
    return analyzer.run_complete_analysis()

if __name__ == "__main__":
    analyzer = main()