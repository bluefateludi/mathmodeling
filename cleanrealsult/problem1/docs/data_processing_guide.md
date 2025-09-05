# 数据处理流程说明

## 1. 数据源概述

### 1.1 原始数据
- **文件名**：`final_cleaned_data.xlsx`
- **位置**：`d:/Program code/pythonproject/mathmodel/final_cleaned_data.xlsx`
- **格式**：Excel文件
- **记录数**：1070条
- **变量数**：31个

### 1.2 数据特征
- **数据类型**：NIPT检测数据
- **时间范围**：涵盖不同孕周的检测结果
- **人群特征**：不同年龄、BMI的孕妇群体
- **检测指标**：包含染色体浓度、Z值等多项指标

## 2. 数据加载流程

### 2.1 环境准备

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
```

### 2.2 数据读取

```python
def load_data(file_path):
    """
    加载Excel数据文件
    
    Args:
        file_path (str): 数据文件路径
    
    Returns:
        pd.DataFrame: 加载的数据
    """
    try:
        data = pd.read_excel(file_path)
        print(f"数据加载成功，共{len(data)}条记录")
        print(f"数据列名：{list(data.columns)}")
        return data
    except Exception as e:
        print(f"数据加载失败：{e}")
        return None
```

### 2.3 数据结构检查

```python
def check_data_structure(data):
    """
    检查数据结构和基本信息
    """
    print("\n=== 数据基本信息 ===")
    print(f"数据形状：{data.shape}")
    print(f"数据类型：\n{data.dtypes}")
    print(f"\n内存使用：{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

## 3. 数据质量评估

### 3.1 缺失值分析

```python
def analyze_missing_values(data):
    """
    分析缺失值情况
    """
    missing_stats = pd.DataFrame({
        '缺失数量': data.isnull().sum(),
        '缺失比例': data.isnull().sum() / len(data) * 100
    })
    
    missing_stats = missing_stats[missing_stats['缺失数量'] > 0]
    missing_stats = missing_stats.sort_values('缺失比例', ascending=False)
    
    print("\n=== 缺失值统计 ===")
    if len(missing_stats) > 0:
        print(missing_stats)
    else:
        print("无缺失值")
    
    return missing_stats
```

### 3.2 数据类型检查

```python
def check_data_types(data):
    """
    检查和转换数据类型
    """
    print("\n=== 数据类型分析 ===")
    
    # 数值型变量
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    print(f"数值型变量（{len(numeric_cols)}个）：{list(numeric_cols)}")
    
    # 分类型变量
    categorical_cols = data.select_dtypes(include=['object']).columns
    print(f"分类型变量（{len(categorical_cols)}个）：{list(categorical_cols)}")
    
    # 日期型变量
    datetime_cols = data.select_dtypes(include=['datetime']).columns
    print(f"日期型变量（{len(datetime_cols)}个）：{list(datetime_cols)}")
    
    return {
        'numeric': list(numeric_cols),
        'categorical': list(categorical_cols),
        'datetime': list(datetime_cols)
    }
```

### 3.3 异常值检测

```python
def detect_outliers(data, columns=None):
    """
    使用IQR方法检测异常值
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    outlier_stats = {}
    
    for col in columns:
        if data[col].dtype in [np.float64, np.int64]:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = outlier_count / len(data) * 100
            
            outlier_stats[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    print("\n=== 异常值检测结果 ===")
    for col, stats in outlier_stats.items():
        if stats['count'] > 0:
            print(f"{col}: {stats['count']}个异常值 ({stats['percentage']:.2f}%)")
    
    return outlier_stats
```

## 4. 变量识别与选择

### 4.1 目标变量识别

```python
def identify_target_variable(data):
    """
    识别Y染色体相关的目标变量
    """
    target_candidates = []
    
    for col in data.columns:
        if 'Y' in col and ('染色体' in col or '浓度' in col or 'Z值' in col):
            target_candidates.append(col)
    
    print("\n=== 目标变量候选 ===")
    for i, col in enumerate(target_candidates):
        print(f"{i+1}. {col}")
        print(f"   数据类型: {data[col].dtype}")
        print(f"   非空值: {data[col].count()}")
        print(f"   描述统计: 均值={data[col].mean():.4f}, 标准差={data[col].std():.4f}")
        print()
    
    return target_candidates
```

### 4.2 特征变量识别

```python
def identify_feature_variables(data):
    """
    识别孕周、BMI、年龄等特征变量
    """
    feature_mapping = {
        '孕周': [],
        'BMI': [],
        '年龄': [],
        '其他': []
    }
    
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64]:
            if '孕周' in col or '周' in col:
                feature_mapping['孕周'].append(col)
            elif 'BMI' in col or 'bmi' in col.lower():
                feature_mapping['BMI'].append(col)
            elif '年龄' in col or 'age' in col.lower():
                feature_mapping['年龄'].append(col)
            else:
                feature_mapping['其他'].append(col)
    
    print("\n=== 特征变量分类 ===")
    for category, variables in feature_mapping.items():
        if variables:
            print(f"{category}: {variables}")
    
    return feature_mapping
```

## 5. 数据预处理

### 5.1 缺失值处理

```python
def handle_missing_values(data, strategy='drop'):
    """
    处理缺失值
    
    Args:
        data: 原始数据
        strategy: 处理策略 ('drop', 'mean', 'median', 'mode')
    """
    if strategy == 'drop':
        # 删除含有缺失值的行
        cleaned_data = data.dropna()
        print(f"删除缺失值后，剩余{len(cleaned_data)}条记录")
    
    elif strategy == 'mean':
        # 用均值填充数值型变量
        cleaned_data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].mean(), inplace=True)
    
    elif strategy == 'median':
        # 用中位数填充数值型变量
        cleaned_data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_data[col].fillna(data[col].median(), inplace=True)
    
    return cleaned_data
```

### 5.2 数据标准化

```python
def standardize_data(data, columns=None):
    """
    对数值型变量进行标准化
    """
    from sklearn.preprocessing import StandardScaler
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    scaler = StandardScaler()
    
    standardized_data[columns] = scaler.fit_transform(data[columns])
    
    print(f"已对{len(columns)}个变量进行标准化")
    return standardized_data, scaler
```

## 6. 数据验证

### 6.1 数据一致性检查

```python
def validate_data_consistency(data):
    """
    检查数据一致性
    """
    issues = []
    
    # 检查年龄合理性
    if '年龄' in data.columns:
        age_col = '年龄'
    else:
        age_cols = [col for col in data.columns if '年龄' in col]
        age_col = age_cols[0] if age_cols else None
    
    if age_col:
        invalid_ages = data[(data[age_col] < 15) | (data[age_col] > 50)]
        if len(invalid_ages) > 0:
            issues.append(f"发现{len(invalid_ages)}个异常年龄值")
    
    # 检查孕周合理性
    week_cols = [col for col in data.columns if '孕周' in col]
    if week_cols:
        week_col = week_cols[0]
        invalid_weeks = data[(data[week_col] < 10) | (data[week_col] > 42)]
        if len(invalid_weeks) > 0:
            issues.append(f"发现{len(invalid_weeks)}个异常孕周值")
    
    # 检查BMI合理性
    bmi_cols = [col for col in data.columns if 'BMI' in col]
    if bmi_cols:
        bmi_col = bmi_cols[0]
        invalid_bmi = data[(data[bmi_col] < 15) | (data[bmi_col] > 45)]
        if len(invalid_bmi) > 0:
            issues.append(f"发现{len(invalid_bmi)}个异常BMI值")
    
    print("\n=== 数据一致性检查 ===")
    if issues:
        for issue in issues:
            print(f"⚠️ {issue}")
    else:
        print("✅ 数据一致性检查通过")
    
    return issues
```

### 6.2 数据完整性验证

```python
def validate_data_completeness(data, required_columns):
    """
    验证必需列是否存在且完整
    """
    missing_columns = []
    incomplete_columns = []
    
    for col in required_columns:
        if col not in data.columns:
            missing_columns.append(col)
        elif data[col].isnull().any():
            incomplete_columns.append(col)
    
    print("\n=== 数据完整性验证 ===")
    if missing_columns:
        print(f"❌ 缺失列: {missing_columns}")
    if incomplete_columns:
        print(f"⚠️ 不完整列: {incomplete_columns}")
    if not missing_columns and not incomplete_columns:
        print("✅ 数据完整性验证通过")
    
    return len(missing_columns) == 0 and len(incomplete_columns) == 0
```

## 7. 数据导出

### 7.1 处理后数据保存

```python
def save_processed_data(data, output_path, format='excel'):
    """
    保存处理后的数据
    """
    try:
        if format == 'excel':
            data.to_excel(output_path, index=False)
        elif format == 'csv':
            data.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"处理后数据已保存至: {output_path}")
        return True
    except Exception as e:
        print(f"数据保存失败: {e}")
        return False
```

### 7.2 数据处理报告

```python
def generate_processing_report(original_data, processed_data, output_path):
    """
    生成数据处理报告
    """
    report = []
    report.append("# 数据处理报告\n")
    
    report.append("## 处理前后对比\n")
    report.append(f"- 原始记录数: {len(original_data)}")
    report.append(f"- 处理后记录数: {len(processed_data)}")
    report.append(f"- 记录保留率: {len(processed_data)/len(original_data)*100:.2f}%")
    report.append(f"- 原始变量数: {len(original_data.columns)}")
    report.append(f"- 处理后变量数: {len(processed_data.columns)}\n")
    
    report.append("## 数据质量指标\n")
    report.append(f"- 缺失值总数: {processed_data.isnull().sum().sum()}")
    report.append(f"- 数据完整性: {(1-processed_data.isnull().sum().sum()/(len(processed_data)*len(processed_data.columns)))*100:.2f}%\n")
    
    report_text = "\n".join(report)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"数据处理报告已保存至: {output_path}")
    return report_text
```

## 8. 完整处理流程

```python
def complete_data_processing_pipeline(input_path, output_dir):
    """
    完整的数据处理流程
    """
    print("开始数据处理流程...")
    
    # 1. 加载数据
    data = load_data(input_path)
    if data is None:
        return False
    
    original_data = data.copy()
    
    # 2. 数据质量评估
    check_data_structure(data)
    analyze_missing_values(data)
    check_data_types(data)
    detect_outliers(data)
    
    # 3. 变量识别
    target_vars = identify_target_variable(data)
    feature_vars = identify_feature_variables(data)
    
    # 4. 数据预处理
    processed_data = handle_missing_values(data, strategy='drop')
    
    # 5. 数据验证
    validate_data_consistency(processed_data)
    
    # 6. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后数据
    processed_data_path = os.path.join(output_dir, 'processed_data.xlsx')
    save_processed_data(processed_data, processed_data_path)
    
    # 生成处理报告
    report_path = os.path.join(output_dir, 'data_processing_report.md')
    generate_processing_report(original_data, processed_data, report_path)
    
    print("数据处理流程完成！")
    return True
```

## 9. 使用示例

```python
if __name__ == "__main__":
    input_file = "d:/Program code/pythonproject/mathmodel/final_cleaned_data.xlsx"
    output_directory = "d:/Program code/pythonproject/mathmodel/cleanrealsult/problem1/data"
    
    # 运行完整处理流程
    success = complete_data_processing_pipeline(input_file, output_directory)
    
    if success:
        print("数据处理成功完成！")
    else:
        print("数据处理失败！")
```

## 10. 注意事项

### 10.1 数据安全
- 确保数据文件路径正确
- 备份原始数据
- 注意数据隐私保护

### 10.2 性能优化
- 大数据集使用分块处理
- 合理设置内存使用限制
- 优化数据类型以节省内存

### 10.3 错误处理
- 添加异常捕获机制
- 提供详细的错误信息
- 实现数据恢复功能

---

**文档版本**: 1.0  
**最后更新**: 2025年  
**维护者**: 数学建模团队