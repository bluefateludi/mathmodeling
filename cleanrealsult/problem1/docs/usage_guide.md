# 使用指南

## 1. 环境准备

### 1.1 Python环境要求
- Python 3.8 或更高版本
- 推荐使用 Python 3.9 或 3.10

### 1.2 依赖包安装

在项目根目录下运行以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

或者逐个安装主要依赖包：

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl statsmodels
```

### 1.3 数据文件准备

确保数据文件 `final_cleaned_data.xlsx` 位于正确路径：
```
d:/Program code/pythonproject/mathmodel/final_cleaned_data.xlsx
```

## 2. 快速开始

### 2.1 运行完整分析

在项目根目录下运行：

```bash
python main.py
```

这将执行完整的分析流程，包括：
- 数据加载和预处理
- 相关性分析
- 多元回归建模
- 结果可视化
- 报告生成

### 2.2 查看结果

分析完成后，结果将保存在以下位置：

- **核心结论文档**: `results/executive_summary.md` 📋
  - 简明扼要的研究结论和关键发现
  - 适合快速了解项目核心要点
- **详细分析报告**: `results/analysis_report.md`
- **相关性热力图（优化版本）**: `results/correlation_analysis_plots_heatmap.png`
  - 移除数字标注，提升视觉清晰度
  - 使用直观的颜色映射和图例设计
  - 红色表示正相关，蓝色表示负相关
- **分析图表组合**: `results/correlation_analysis_plots_analysis.png`
- **完整图表**: `results/correlation_analysis_plots.png`（向后兼容）
- **详细日志**: 控制台输出

## 3. 项目结构说明

```
problem1/
├── README.md                    # 项目概述和主要发现
├── main.py                      # 主执行脚本
├── requirements.txt             # 依赖包列表
├── code/                        # 源代码目录
│   └── correlation_analysis.py  # 核心分析模块
├── docs/                        # 文档目录
│   ├── methodology.md           # 方法论说明
│   ├── data_processing_guide.md # 数据处理流程
│   └── usage_guide.md           # 使用指南（本文件）
├── results/                     # 结果输出目录
│   ├── analysis_report.md       # 分析报告
│   └── correlation_analysis_plots.png # 可视化图表
└── plots/                       # 图表临时目录
```

## 4. 核心功能模块

### 4.1 数据加载模块

```python
from code.correlation_analysis import NIPTCorrelationAnalysis

# 创建分析实例
analyzer = NIPTCorrelationAnalysis()

# 加载数据
data = analyzer.load_data('path/to/data.xlsx')
```

### 4.2 相关性分析

```python
# 执行相关性分析
corr_results = analyzer.correlation_analysis(data)
print(corr_results)
```

### 4.3 回归建模

```python
# 建立多元回归模型
model_results = analyzer.regression_analysis(data)
print(f"R²: {model_results['r2']:.4f}")
print(f"RMSE: {model_results['rmse']:.4f}")
```

### 4.4 可视化生成

```python
# 生成可视化图表
analyzer.create_visualizations(data, 'output_path.png')
```

## 5. 自定义分析

### 5.1 修改分析参数

可以通过修改 `main.py` 中的参数来自定义分析：

```python
# 修改数据文件路径
data_file = "your/custom/path/data.xlsx"

# 修改输出目录
results_dir = "your/custom/results/"
plots_dir = "your/custom/plots/"
```

### 5.2 添加新的分析变量

在 `correlation_analysis.py` 中修改变量列表：

```python
# 在 correlation_analysis 方法中添加新变量
target_vars = ['Y染色体的Z值', 'your_new_variable']
feature_vars = ['孕周', 'BMI', 'your_new_feature']
```

### 5.3 自定义可视化

可以修改 `create_visualizations` 方法来添加新的图表类型：

```python
def create_custom_plot(self, data):
    plt.figure(figsize=(10, 6))
    # 添加自定义绘图代码
    plt.savefig('custom_plot.png', dpi=300, bbox_inches='tight')
```

## 6. 常见问题解决

### 6.1 数据文件找不到

**错误信息**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决方案**:
1. 检查数据文件路径是否正确
2. 确认文件名拼写无误
3. 使用绝对路径而非相对路径

```python
# 使用绝对路径
data_file = r"d:\Program code\pythonproject\mathmodel\final_cleaned_data.xlsx"
```

### 6.2 依赖包缺失

**错误信息**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:
```bash
pip install xxx
# 或者重新安装所有依赖
pip install -r requirements.txt
```

### 6.3 内存不足

**错误信息**: `MemoryError`

**解决方案**:
1. 关闭其他程序释放内存
2. 使用数据分块处理
3. 减少可视化图表的分辨率

```python
# 降低图表分辨率
plt.savefig('plot.png', dpi=150)  # 默认300改为150
```

### 6.4 中文显示问题

**问题**: 图表中中文显示为方块

**解决方案**:
```python
# 在代码开头添加
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
```

## 7. 性能优化建议

### 7.1 数据处理优化

```python
# 使用更高效的数据类型
data = data.astype({
    'int_column': 'int32',    # 而非int64
    'float_column': 'float32' # 而非float64
})

# 删除不需要的列
data = data.drop(['unnecessary_column'], axis=1)
```

### 7.2 可视化优化

```python
# 减少数据点数量（适用于大数据集）
sampled_data = data.sample(n=1000)  # 随机采样1000个点

# 使用更高效的绘图后端
import matplotlib
matplotlib.use('Agg')  # 不显示图形，只保存
```

### 7.3 并行处理

```python
# 对于大规模计算，可以使用并行处理
from multiprocessing import Pool

def parallel_analysis(data_chunk):
    # 分析代码
    return results

# 分割数据并并行处理
with Pool() as pool:
    results = pool.map(parallel_analysis, data_chunks)
```

## 8. 扩展功能

### 8.1 添加新的统计检验

```python
from scipy import stats

# 添加正态性检验
def normality_test(data, column):
    statistic, p_value = stats.shapiro(data[column])
    return {'statistic': statistic, 'p_value': p_value}
```

### 8.2 模型诊断

```python
# 添加残差分析
def residual_analysis(model, X, y):
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # 绘制残差图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差分析')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
```

### 8.3 交互式可视化

```python
# 使用plotly创建交互式图表
import plotly.express as px

def create_interactive_plot(data):
    fig = px.scatter(data, x='孕周', y='Y染色体的Z值', 
                    color='BMI', title='交互式散点图')
    fig.write_html('interactive_plot.html')
```

## 9. 版本控制

### 9.1 Git使用建议

```bash
# 初始化git仓库
git init

# 添加.gitignore文件
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "results/*.png" >> .gitignore

# 提交代码
git add .
git commit -m "Initial commit: NIPT correlation analysis"
```

### 9.2 版本标记

```bash
# 创建版本标签
git tag -a v1.0 -m "Version 1.0: Basic correlation analysis"
git tag -a v1.1 -m "Version 1.1: Added visualization improvements"
```

## 10. 技术支持

### 10.1 日志记录

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("分析开始")
```

### 10.2 错误报告

如果遇到问题，请提供以下信息：
1. Python版本
2. 依赖包版本
3. 完整的错误信息
4. 数据文件基本信息（行数、列数、文件大小）
5. 运行环境（操作系统、内存大小）

### 10.3 联系方式

- 项目维护者：数学建模团队
- 更新日期：2025年
- 文档版本：1.0

---

**注意**: 本指南会根据项目发展持续更新，请定期查看最新版本。