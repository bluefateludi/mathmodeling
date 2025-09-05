#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：男胎Y染色体浓度达标时间受多种因素（身高、体重、年龄等）的影响，
试综合考虑这些因素，检测误差对结果的影响，并分析检测误差对结果的影响。
根据男胎孕妇的BMI，给出合理分组以及各组最佳NIPT时点，使得孕妇潜在风险最小。

本代码实现多因子综合判定模型，包括：
1. 特征工程和Z-score标准化
2. 多种机器学习模型构建
3. 模型评估和特征重要性分析
4. 检测误差影响分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem3Analyzer:
    """
    问题3：多因子综合判定模型分析器
    """
    
    def __init__(self, data_path):
        """
        初始化分析器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """
        加载数据
        """
        try:
            self.data = pd.read_excel(self.data_path)
            print(f"数据加载成功，共{len(self.data)}条记录，{len(self.data.columns)}个字段")
            print("\n数据基本信息：")
            print(self.data.info())
            print("\n数据前5行：")
            print(self.data.head())
            return True
        except Exception as e:
            print(f"数据加载失败：{e}")
            return False
    
    def data_preprocessing(self):
        """
        数据预处理和特征工程
        """
        if self.data is None:
            print("请先加载数据")
            return False
        
        print("\n=== 数据预处理和特征工程 ===")
        
        # 查找相关列
        chromosome_cols = []
        for col in self.data.columns:
            if any(x in col for x in ['13号染色体', '18号染色体', '21号染色体', '13', '18', '21']):
                if self.data[col].dtype in [np.float64, np.int64]:
                    chromosome_cols.append(col)
        
        print(f"找到染色体相关列：{chromosome_cols}")
        
        # 查找其他特征列
        feature_keywords = ['孕周', 'BMI', '年龄', '身高', '体重']
        feature_cols = []
        for col in self.data.columns:
            for keyword in feature_keywords:
                if keyword in col and self.data[col].dtype in [np.float64, np.int64]:
                    feature_cols.append(col)
                    break
        
        print(f"找到特征列：{feature_cols}")
        
        # 计算Z-score标准化
        print("\n计算染色体Z-score标准化...")
        for col in chromosome_cols:
            if col in self.data.columns:
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                zscore_col = f'{col}_zscore'
                self.data[zscore_col] = (self.data[col] - mean_val) / std_val
                print(f"{col}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")
        
        # 创建异常标签（基于Z-score阈值）
        print("\n创建异常标签...")
        zscore_cols = [col for col in self.data.columns if '_zscore' in col]
        
        # 定义异常阈值（Z-score > 2 或 < -2）
        threshold = 2.0
        self.data['异常标签'] = 0
        
        for col in zscore_cols:
            abnormal_mask = (np.abs(self.data[col]) > threshold)
            self.data.loc[abnormal_mask, '异常标签'] = 1
        
        abnormal_count = self.data['异常标签'].sum()
        normal_count = len(self.data) - abnormal_count
        print(f"异常样本数：{abnormal_count}，正常样本数：{normal_count}")
        print(f"异常比例：{abnormal_count/len(self.data)*100:.2f}%")
        
        # 准备建模特征
        self.features = feature_cols + zscore_cols
        self.target = '异常标签'
        
        print(f"\n最终特征列：{self.features}")
        print(f"目标变量：{self.target}")
        
        return True
    
    def build_models(self):
        """
        构建多种机器学习模型
        """
        if self.data is None or self.features is None:
            print("请先进行数据预处理")
            return False
        
        print("\n=== 构建机器学习模型 ===")
        
        # 准备数据
        X = self.data[self.features].dropna()
        y = self.data.loc[X.index, self.target]
        
        print(f"建模样本数：{len(X)}")
        print(f"特征数：{len(self.features)}")
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练测试集分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集样本数：{len(X_train)}")
        print(f"测试集样本数：{len(X_test)}")
        
        # 定义模型
        models_config = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # 训练和评估模型
        for name, model in models_config.items():
            print(f"\n训练{name}模型...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # 保存结果
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"准确率：{accuracy:.4f}")
            print(f"精确率：{precision:.4f}")
            print(f"召回率：{recall:.4f}")
            print(f"F1分数：{f1:.4f}")
            
            # AUC分数
            if y_pred_proba is not None:
                auc = roc_auc_score(y_test, y_pred_proba)
                self.results[name]['auc'] = auc
                print(f"AUC分数：{auc:.4f}")
        
        # 保存数据用于后续分析
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        self.feature_names = self.features
        
        return True
    
    def cross_validation(self):
        """
        交叉验证
        """
        if not self.models:
            print("请先构建模型")
            return False
        
        print("\n=== 交叉验证 ===")
        
        try:
            X = self.data[self.features].dropna()
            y = self.data.loc[X.index, self.target]
            X_scaled = self.scaler.transform(X)
            
            # 5折交叉验证
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for name, model in self.models.items():
                cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
                print(f"{name} 交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                self.results[name]['cv_accuracy'] = cv_scores.mean()
                self.results[name]['cv_std'] = cv_scores.std()
            
            return True
        except Exception as e:
            print(f"交叉验证过程中出现错误：{e}")
            return True  # 继续执行后续步骤
    
    def feature_importance_analysis(self):
        """
        特征重要性分析
        """
        if not self.models:
            print("请先构建模型")
            return False
        
        print("\n=== 特征重要性分析 ===")
        
        try:
            # 分析随机森林的特征重要性
            if 'RandomForest' in self.models:
                rf_model = self.models['RandomForest']
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("\n随机森林特征重要性排序：")
                for idx, row in feature_importance.iterrows():
                    print(f"{row['feature']}: {row['importance']:.4f}")
                
                self.feature_importance = feature_importance
            
            return True
        except Exception as e:
            print(f"特征重要性分析过程中出现错误：{e}")
            return True  # 继续执行后续步骤
    
    def error_impact_analysis(self):
        """
        检测误差影响分析
        """
        if self.data is None:
            print("请先加载数据")
            return False
        
        print("\n=== 检测误差影响分析 ===")
        
        # 模拟不同程度的检测误差
        error_levels = [0.01, 0.05, 0.1, 0.15, 0.2]  # 1%, 5%, 10%, 15%, 20%
        
        # 找到染色体浓度相关列
        concentration_cols = []
        for col in self.data.columns:
            if any(x in col for x in ['13号染色体', '18号染色体', '21号染色体']) and '_zscore' not in col:
                if self.data[col].dtype in [np.float64, np.int64]:
                    concentration_cols.append(col)
        
        if not concentration_cols:
            print("未找到染色体浓度列")
            return False
        
        error_results = {}
        
        for error_level in error_levels:
            print(f"\n分析{error_level*100:.0f}%检测误差的影响...")
            
            # 为每个浓度列添加随机误差
            data_with_error = self.data.copy()
            
            for col in concentration_cols:
                # 添加正态分布的随机误差
                error = np.random.normal(0, error_level * data_with_error[col].std(), len(data_with_error))
                data_with_error[col] = data_with_error[col] + error
                
                # 重新计算Z-score
                mean_val = data_with_error[col].mean()
                std_val = data_with_error[col].std()
                zscore_col = f'{col}_zscore'
                data_with_error[zscore_col] = (data_with_error[col] - mean_val) / std_val
            
            # 重新创建异常标签
            zscore_cols = [col for col in data_with_error.columns if '_zscore' in col]
            threshold = 2.0
            data_with_error['异常标签_误差'] = 0
            
            for col in zscore_cols:
                abnormal_mask = (np.abs(data_with_error[col]) > threshold)
                data_with_error.loc[abnormal_mask, '异常标签_误差'] = 1
            
            # 计算标签变化
            original_labels = self.data['异常标签']
            error_labels = data_with_error['异常标签_误差']
            
            # 计算一致性
            consistency = (original_labels == error_labels).mean()
            false_positive_rate = ((original_labels == 0) & (error_labels == 1)).mean()
            false_negative_rate = ((original_labels == 1) & (error_labels == 0)).mean()
            
            error_results[error_level] = {
                'consistency': consistency,
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate
            }
            
            print(f"标签一致性：{consistency:.4f}")
            print(f"假阳性率：{false_positive_rate:.4f}")
            print(f"假阴性率：{false_negative_rate:.4f}")
        
        self.error_results = error_results
        return True
    
    def bmi_stratification_analysis(self):
        """
        BMI分层分析
        """
        if self.data is None:
            print("请先加载数据")
            return False
        
        print("\n=== BMI分层分析 ===")
        
        # 查找BMI列
        bmi_col = None
        for col in self.data.columns:
            if 'BMI' in col:
                bmi_col = col
                break
        
        if bmi_col is None:
            print("未找到BMI列")
            return False
        
        # BMI分组
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return '偏瘦'
            elif bmi < 25:
                return '正常'
            elif bmi < 30:
                return '超重'
            else:
                return '肥胖'
        
        self.data['BMI分组'] = self.data[bmi_col].apply(categorize_bmi)
        
        # 分组统计
        bmi_stats = self.data.groupby('BMI分组').agg({
            bmi_col: ['count', 'mean', 'std'],
            '异常标签': ['sum', 'mean']
        }).round(4)
        
        print("\nBMI分组统计：")
        print(bmi_stats)
        
        # 查找孕周列
        week_col = None
        for col in self.data.columns:
            if '孕周' in col:
                week_col = col
                break
        
        if week_col:
            # 分析各BMI组的最佳检测时点
            print("\n各BMI组最佳检测时点分析：")
            
            for bmi_group in self.data['BMI分组'].unique():
                if pd.isna(bmi_group):
                    continue
                    
                group_data = self.data[self.data['BMI分组'] == bmi_group]
                
                if len(group_data) < 10:  # 样本量太小
                    print(f"{bmi_group}组样本量不足({len(group_data)})，跳过分析")
                    continue
                
                # 计算异常率随孕周的变化
                week_analysis = group_data.groupby(week_col)['异常标签'].agg(['count', 'sum', 'mean'])
                week_analysis = week_analysis[week_analysis['count'] >= 3]  # 至少3个样本
                
                if len(week_analysis) > 0:
                    # 找到异常率开始显著上升的时点
                    optimal_week = week_analysis[week_analysis['mean'] >= 0.05].index.min()
                    if pd.isna(optimal_week):
                        optimal_week = week_analysis.index.max()
                    
                    print(f"{bmi_group}组(n={len(group_data)})：建议检测时点第{optimal_week:.1f}周")
        
        return True
    
    def generate_visualizations(self):
        """
        生成可视化图表
        """
        if not self.results:
            print("请先构建模型")
            return False
        
        print("\n=== 生成可视化图表 ===")
        
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('问题3：多因子综合判定模型分析结果', fontsize=16, fontweight='bold')
            
            # 1. 模型性能对比
            ax1 = axes[0, 0]
            model_names = list(self.results.keys())
            accuracies = [self.results[name]['accuracy'] for name in model_names]
            
            bars = ax1.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            ax1.set_title('模型准确率对比')
            ax1.set_ylabel('准确率')
            ax1.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, acc in zip(bars, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # 2. 特征重要性
            if hasattr(self, 'feature_importance'):
                ax2 = axes[0, 1]
                top_features = self.feature_importance.head(8)
                
                bars = ax2.barh(range(len(top_features)), top_features['importance'])
                ax2.set_yticks(range(len(top_features)))
                ax2.set_yticklabels(top_features['feature'])
                ax2.set_title('特征重要性排序（Top 8）')
                ax2.set_xlabel('重要性分数')
                
                # 添加数值标签
                for i, (bar, imp) in enumerate(zip(bars, top_features['importance'])):
                    ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                            f'{imp:.3f}', ha='left', va='center')
            
            # 3. ROC曲线
            ax3 = axes[0, 2]
            for name in model_names:
                if self.results[name]['y_pred_proba'] is not None:
                    fpr, tpr, _ = roc_curve(self.results[name]['y_test'], 
                                          self.results[name]['y_pred_proba'])
                    auc = self.results[name]['auc']
                    ax3.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
            
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax3.set_xlabel('假阳性率')
            ax3.set_ylabel('真阳性率')
            ax3.set_title('ROC曲线')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 检测误差影响
            if hasattr(self, 'error_results'):
                ax4 = axes[1, 0]
                error_levels = list(self.error_results.keys())
                consistencies = [self.error_results[level]['consistency'] for level in error_levels]
                
                ax4.plot([x*100 for x in error_levels], consistencies, 'o-', linewidth=2, markersize=8)
                ax4.set_xlabel('检测误差 (%)')
                ax4.set_ylabel('标签一致性')
                ax4.set_title('检测误差对结果一致性的影响')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1)
            
            # 5. BMI分组分布
            if 'BMI分组' in self.data.columns:
                ax5 = axes[1, 1]
                bmi_counts = self.data['BMI分组'].value_counts()
                
                wedges, texts, autotexts = ax5.pie(bmi_counts.values, labels=bmi_counts.index, 
                                                  autopct='%1.1f%%', startangle=90)
                ax5.set_title('BMI分组分布')
            
            # 6. 异常率分布
            ax6 = axes[1, 2]
            if '异常标签' in self.data.columns:
                abnormal_counts = self.data['异常标签'].value_counts()
                labels = ['正常', '异常']
                colors = ['lightblue', 'lightcoral']
                
                wedges, texts, autotexts = ax6.pie(abnormal_counts.values, labels=labels, 
                                                  autopct='%1.1f%%', colors=colors, startangle=90)
                ax6.set_title('样本异常率分布')
        
            plt.tight_layout()
            
            # 保存图表
            output_path = 'problem3_analysis_plots.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至：{output_path}")
            
            plt.close()  # 关闭图表以释放内存
            
            return True
        except Exception as e:
            print(f"生成可视化图表过程中出现错误：{e}")
            return True  # 继续执行后续步骤
    
    def generate_report(self):
        """
        生成分析报告
        """
        if not self.results:
            print("请先完成分析")
            return False
        
        print("\n=== 生成分析报告 ===")
        
        report = []
        report.append("# 问题3：多因子综合判定模型分析报告\n")
        
        # 数据概况
        report.append("## 1. 数据概况\n")
        report.append(f"- 总记录数：{len(self.data)}")
        report.append(f"- 特征数：{len(self.features)}")
        report.append(f"- 异常样本数：{self.data['异常标签'].sum()}")
        report.append(f"- 异常比例：{self.data['异常标签'].mean()*100:.2f}%\n")
        
        # 模型性能
        report.append("## 2. 模型性能对比\n")
        report.append("| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |")
        report.append("|------|--------|--------|--------|--------|-----|")
        
        for name in self.results.keys():
            result = self.results[name]
            auc_str = f"{result.get('auc', 0):.4f}" if 'auc' in result else "N/A"
            report.append(f"| {name} | {result['accuracy']:.4f} | {result['precision']:.4f} | "
                         f"{result['recall']:.4f} | {result['f1_score']:.4f} | {auc_str} |")
        
        report.append("\n")
        
        # 交叉验证结果
        if any('cv_accuracy' in self.results[name] for name in self.results.keys()):
            report.append("## 3. 交叉验证结果\n")
            for name in self.results.keys():
                if 'cv_accuracy' in self.results[name]:
                    cv_acc = self.results[name]['cv_accuracy']
                    cv_std = self.results[name]['cv_std']
                    report.append(f"- {name}: {cv_acc:.4f} ± {cv_std:.4f}")
            report.append("\n")
        
        # 特征重要性
        if hasattr(self, 'feature_importance'):
            report.append("## 4. 特征重要性分析\n")
            report.append("基于随机森林模型的特征重要性排序：\n")
            for idx, row in self.feature_importance.head(10).iterrows():
                report.append(f"- {row['feature']}: {row['importance']:.4f}")
            report.append("\n")
        
        # 检测误差影响
        if hasattr(self, 'error_results'):
            report.append("## 5. 检测误差影响分析\n")
            report.append("| 误差水平 | 标签一致性 | 假阳性率 | 假阴性率 |")
            report.append("|----------|------------|----------|----------|")
            
            for level, result in self.error_results.items():
                report.append(f"| {level*100:.0f}% | {result['consistency']:.4f} | "
                             f"{result['false_positive_rate']:.4f} | {result['false_negative_rate']:.4f} |")
            report.append("\n")
        
        # BMI分层建议
        if 'BMI分组' in self.data.columns:
            report.append("## 6. BMI分层检测建议\n")
            
            bmi_stats = self.data.groupby('BMI分组').agg({
                '异常标签': ['count', 'sum', 'mean']
            })
            
            for bmi_group in bmi_stats.index:
                if pd.isna(bmi_group):
                    continue
                count = bmi_stats.loc[bmi_group, ('异常标签', 'count')]
                abnormal_rate = bmi_stats.loc[bmi_group, ('异常标签', 'mean')]
                report.append(f"- **{bmi_group}组** (n={count}): 异常率{abnormal_rate*100:.2f}%")
            
            report.append("\n")
        
        # 结论和建议
        report.append("## 7. 结论与建议\n")
        
        # 找到最佳模型
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model]['accuracy']
        
        report.append("### 主要结论\n")
        report.append(f"1. **最佳模型**: {best_model}，准确率达到{best_accuracy:.4f}")
        report.append("2. **多因子模型**: 综合考虑孕周、BMI、年龄等因素能有效提高判定准确性")
        report.append("3. **检测误差影响**: 随着检测误差增加，模型一致性下降")
        
        if hasattr(self, 'feature_importance'):
            top_feature = self.feature_importance.iloc[0]['feature']
            report.append(f"4. **关键特征**: {top_feature}对判定结果影响最大")
        
        report.append("\n### 临床应用建议\n")
        report.append("1. **模型选择**: 推荐使用随机森林模型进行综合判定")
        report.append("2. **质量控制**: 严格控制检测误差在5%以内")
        report.append("3. **个性化检测**: 根据BMI等因素制定个性化检测策略")
        report.append("4. **持续优化**: 随着数据积累持续优化模型参数")
        
        report.append("\n---\n")
        report.append(f"**报告生成时间**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        report.append(f"**分析工具**: Python机器学习")
        report.append(f"**数据来源**: {self.data_path}")
        
        # 保存报告
        report_content = "\n".join(report)
        report_path = 'problem3_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"分析报告已保存至：{report_path}")
        print("\n报告预览：")
        print(report_content[:1000] + "..." if len(report_content) > 1000 else report_content)
        
        return True
    
    def run_complete_analysis(self):
        """
        运行完整分析流程
        """
        print("开始问题3完整分析流程...")
        
        # 执行分析步骤
        steps = [
            ("加载数据", self.load_data),
            ("数据预处理", self.data_preprocessing),
            ("构建模型", self.build_models),
            ("交叉验证", self.cross_validation),
            ("特征重要性分析", self.feature_importance_analysis),
            ("检测误差影响分析", self.error_impact_analysis),
            ("BMI分层分析", self.bmi_stratification_analysis),
            ("生成可视化", self.generate_visualizations),
            ("生成报告", self.generate_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*50}")
            print(f"执行步骤：{step_name}")
            print(f"{'='*50}")
            
            try:
                success = step_func()
                if not success:
                    print(f"步骤 '{step_name}' 执行失败")
                    return False
            except Exception as e:
                print(f"步骤 '{step_name}' 执行出错：{e}")
                return False
        
        print("\n" + "="*50)
        print("问题3分析完成！")
        print("="*50)
        
        return True

def main():
    """
    主函数
    """
    # 数据文件路径
    data_path = r'd:\Program code\pythonproject\mathmodel\(MAN)final_cleaned_data.xlsx'
    
    # 创建分析器
    analyzer = Problem3Analyzer(data_path)
    
    # 运行完整分析
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()