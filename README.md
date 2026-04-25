# 面向工业时间序列的自主分析智能体系统
## Autonomous Agent System for Industrial Time Series Analysis

### 1. 研究目标
本项目面向工业场景中的多变量时间序列数据，构建一套可复现、可扩展的多智能体分析系统。系统以大语言模型承担任务规划与策略决策，以本地Python算法库承担数值计算与模型训练，目标是形成从数据清洗到预测评估的端到端自动化流程。

### 2. 方法框架
系统基于 LangGraph 构建五节点流水线：
- Preprocess Agent：缺失值处理、异常值控制、数据质量分析。
- Analysis Agent：趋势、季节性、平稳性及潜在风险分析。
- Validation Agent：候选模型筛选、超参数搜索与验证集评估。
- Forecast Agent：单模型预测、加权集成预测、测试集指标计算。
- Report Agent：生成结构化实验总结与结果报告。

### 3. 关键设计
- 决策与计算解耦：LLM不直接处理高维矩阵运算，只负责任务规划。
- 本地函数执行：统计、机器学习与预测模型均在本地执行，保证结果可验证。
- 可控自动调参：支持组合数上限与耐心值早停，避免无约束试错。
- 统一模型工厂：通过 llm_factory 统一初始化 OpenAI/Anthropic 模型，减少重复代码。

### 4. 环境要求
- Python 3.10+
- 建议使用虚拟环境

安装依赖：

```bash
cd time_series_agent
pip install -r requirements.txt
```

### 5. 运行方式
在仓库根目录执行：

```bash
python time_series_agent/main.py \
  --data dataset/ETTh1.csv \
  --output-dir results_midterm \
  --num-slices 10 \
  --input-length 512 \
  --horizon 96 \
  --k-models 3 \
  --llm-provider anthropic \
  --llm-model claude-3-5-sonnet-20241022
```

常用参数说明：
- --slice-delay-seconds：切片之间延迟秒数，用于缓解API限流。
- --debug：启用图执行流式调试模式。
- --date-column 与 --value-column：指定输入数据时间列与目标列名。

### 6. 环境变量
根据模型提供方配置API Key：

```bash
# Anthropic
set ANTHROPIC_API_KEY=your_key

# OpenAI
set OPENAI_API_KEY=your_key
```

### 7. 输出产物
运行完成后，系统会在 output_dir/reports 下生成：
- complete_time_series_report_时间戳.json：全流程结果（含每个切片中间状态）。
- aggregated_forecast_results_时间戳.json：跨切片聚合后的最终指标与预测结果。

### 8. 参考文献
- Zhao, H., et al. TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis. arXiv:2510.01538.
- Liu, F. T., Ting, K. M., Zhou, Z.-H. Isolation Forest. ICDM 2008.
