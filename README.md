# DataPilot：表格数据自动分析与建模 Agent

DataPilot 是一个面向 CSV/XLSX 表格数据的自动分析项目。用户上传数据集并填写分析目标后，系统会完成数据画像、任务理解、EDA 图表生成、确定性预处理、模型训练、指标评估和 Markdown 报告生成。

这个项目的设计重点是 **Agent 负责理解和规划，机器学习流程由受控 Python Pipeline 执行**。它不是让大模型随意写代码和执行代码，而是把 Agent 放在更可控、更适合工程落地的位置。

## 当前能完成什么

- 上传 CSV、XLSX、XLS 数据集
- 自动生成数据画像：行列数、字段类型、缺失率、重复行、候选目标列
- 根据用户自然语言目标理解分析任务
- 使用 OpenAI Agents SDK 可选增强任务理解
- 自动判断分类或回归任务
- 自动生成 EDA 图表：缺失值、目标分布、相关性热力图
- 自动构建 sklearn 预处理 Pipeline
- 自动训练多个候选模型，包括线性模型、随机森林、Boosting、XGBoost
- 自动选择当前指标下表现最好的模型
- 输出分类/回归评估指标
- 保存清洗后的数据、特征预览和预处理摘要
- 生成分类/回归评估图表
- 生成最佳模型特征重要性
- 支持上传新数据并调用已训练模型进行预测
- 支持离线 Benchmark 评测，统计 pipeline 成功率、目标列识别准确率、任务类型识别准确率和耗时
- 保存模型、图表、上下文和 Markdown 报告
- 网页端默认中文显示，支持中文/英文切换
- 没有 API Key 时自动回退到规则逻辑，项目仍可运行

## Agent 用在哪里

Agent 目前只用于 **工作流规划和任务理解**，代码位置：

```text
src/datapilot/agents/agent_planner.py
```

当 `ENABLE_AGENT_WORKFLOW=true` 且 `.env` 中配置了真实 `OPENAI_API_KEY` 时，系统会把下面这些信息交给 Agent：

- 用户填写的分析目标
- 用户手动选择的目标列，如果有
- 数据集行列规模
- 所有字段名
- 字段类型
- 每列缺失率
- 每列唯一值数量
- 每列前几个样例值
- 前 5 行样例数据
- 规则系统初步识别出的候选目标列

Agent 需要输出结构化计划：

```json
{
  "target_column": "churn",
  "task_type": "classification",
  "metric": "f1_macro",
  "analysis_focus": ["流失影响因素", "预测模型", "业务建议"],
  "workflow_steps": ["profile_dataset", "run_eda", "train_models", "evaluate_model"],
  "confidence": 0.86,
  "reason": "用户目标提到客户流失，字段 churn 是二分类标签。"
}
```

然后系统会做规则校验：

- `target_column` 必须真实存在于数据集中
- `task_type` 必须与目标列的数据类型基本一致
- Agent 不能凭空创造字段
- Agent 失败时自动回退到规则推断

网页端的 `Agent 计划` 面板会显示本次任务理解来源：

- `source=agent`：使用了 Agents SDK
- `source=rules`：使用了规则回退

## Agent 不会用在哪里

当前版本不会让 Agent 做这些事情：

- 不让 Agent 直接执行任意 Python 代码
- 不让 Agent 直接读写本地任意文件
- 不让 Agent 自己决定删除哪些文件
- 不让 Agent 自己训练模型
- 不让 Agent 自己生成 pandas/sklearn 代码并执行
- 不让 Agent 直接修改数据集
- 不让 Agent 绕过规则校验选择不存在的目标列

这些步骤都由确定性代码完成：

```text
src/datapilot/tools/profiling.py        数据画像
src/datapilot/tools/eda.py              EDA 图表
src/datapilot/tools/ml.py               预处理、训练、评估
src/datapilot/tools/task_inference.py   规则回退
```

这样做的原因是：算法工程项目需要可解释、可复现、可调试。Agent 用来提升自然语言理解和流程规划，核心 ML 训练链路保持稳定。

## 完整运行流程

```text
用户上传数据集
  |
  v
读取 CSV/XLSX
  |
  v
Dataset Profiler 生成数据画像
  |
  v
Agent Planner / Rule Fallback 理解任务
  |
  v
规则校验目标列和任务类型
  |
  v
EDA 工具生成图表
  |
  v
sklearn Pipeline 自动预处理特征
  |
  v
训练候选模型
  |
  v
评估模型并选择最佳模型
  |
  v
生成结构化上下文
  |
  v
LLM 可选润色 Markdown 报告
  |
  v
网页端展示指标、图表、Agent 计划和报告
```

## 支持的任务

当前主要支持：

- 二分类
- 多分类
- 回归

分类任务会输出：

- Accuracy
- Precision Macro
- Recall Macro
- F1 Macro

回归任务会输出：

- MAE
- RMSE
- R2

## 候选模型

当前版本会根据任务类型自动训练一组候选模型，并用主指标选择最佳模型。分类任务默认用 `f1_macro` 选择，回归任务默认用 `r2` 选择。

分类候选模型：

- Logistic Regression
- Random Forest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- HistGradientBoosting Classifier
- XGBoost Classifier

回归候选模型：

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor
- XGBoost Regressor

这些模型目前是候选模型池，不做复杂自动调参。某个模型如果因为数据规模、标签格式或依赖兼容问题训练失败，系统会记录失败原因，并继续比较其他可用模型，避免整个分析任务中断。

当前暂不支持：

- 时间序列预测
- 聚类分析
- 深度学习模型
- 自动超参数搜索
- SHAP 可解释性
- 多轮对话式数据分析
- 真正的多 Agent 协作

这些可以作为后续版本继续扩展。

## 技术栈

- Python
- FastAPI
- uv
- pandas
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- OpenAI Python SDK
- OpenAI Agents SDK
- HTML/CSS/JavaScript 原生网页端

## 快速开始

安装依赖：

```bash
uv sync
```

启动服务：

```bash
uv run uvicorn datapilot.main:app --reload
```

打开网页端：

```text
http://127.0.0.1:8000
```

打开 API 文档：

```text
http://127.0.0.1:8000/docs
```

## 配置 API Key

项目根目录需要有 `.env` 文件。本地 `.env` 不会提交到 GitHub。

示例配置：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini
ENABLE_LLM_REPORT=true
ENABLE_AGENT_WORKFLOW=true
AGENT_MODEL=gpt-4o-mini
DATA_DIR=storage
```

你通常只需要修改：

```env
OPENAI_API_KEY=你的真实API_KEY
OPENAI_BASE_URL=你的base_url
AGENT_MODEL=你的Agent模型名
OPENAI_MODEL=你的报告生成模型名
```

配置说明：

- `OPENAI_API_KEY`：你的 API Key
- `OPENAI_BASE_URL`：OpenAI 兼容网关地址；如果直接使用默认 OpenAI 地址，可以留空
- `ENABLE_AGENT_WORKFLOW`：是否启用 Agents SDK 做任务理解和工作流计划
- `AGENT_MODEL`：Agent Planner 使用的模型
- `ENABLE_LLM_REPORT`：是否用 LLM 润色最终报告
- `OPENAI_MODEL`：报告生成使用的模型
- `DATA_DIR`：数据、图表、模型和报告保存目录

如果 `OPENAI_API_KEY` 为空或仍是 `your_api_key_here`，系统不会请求 API，会自动使用规则逻辑。

## API 使用流程

上传数据集：

```bash
curl -F "file=@examples/titanic_sample.csv" http://127.0.0.1:8000/datasets/upload
```

创建分析任务：

```bash
curl -X POST http://127.0.0.1:8000/jobs \
  -H "Content-Type: application/json" \
  -d "{\"dataset_id\":\"<dataset_id>\",\"user_goal\":\"预测乘客是否生还，并分析影响因素\",\"target_column\":\"survived\"}"
```

读取报告：

```bash
curl http://127.0.0.1:8000/jobs/<job_id>/report
```

使用已训练模型预测新数据：

```bash
curl -F "file=@new_data.csv" http://127.0.0.1:8000/jobs/<job_id>/predict
```

预测接口会加载该任务保存的最佳模型 pipeline，对新数据执行同样的预处理，并返回预测结果预览和预测 CSV 路径。

## Benchmark 评测

项目内置了一个不依赖外网的 benchmark 模块，用 sklearn 内置数据集和 `examples/titanic_sample.csv` 评估工作流稳定性。

运行默认 benchmark：

```bash
uv run datapilot-benchmark
```

只跑前两个数据集，适合快速检查：

```bash
uv run datapilot-benchmark --max-cases 2
```

启用 Agent 任务理解一起评测：

```bash
uv run datapilot-benchmark --use-agent
```

默认评测数据集：

- Breast Cancer
- Iris
- Wine
- Diabetes
- Titanic Sample

输出指标：

- pipeline 成功率
- 目标列识别准确率
- 任务类型识别准确率
- 每个数据集最佳模型
- 每个数据集训练耗时
- Agent 使用次数
- 规则回退次数

输出文件：

```text
benchmarks/results/<run_id>/
  benchmark_results.json
  benchmark_results.csv
  benchmark_report.md
```

## 项目结构

```text
src/datapilot/
  benchmarks.py           Benchmark 评测模块
  agents/
    agent_planner.py      Agents SDK 任务理解与计划
    orchestrator.py       主工作流编排
    report_agent.py       报告生成与 LLM 润色
  api/
    datasets.py           数据集上传和画像接口
    jobs.py               分析任务接口
  core/
    config.py             环境变量配置
    storage.py            存储路径工具
  schemas/
    jobs.py               API 输入输出模型
  tools/
    data_loader.py        CSV/XLSX 读取
    profiling.py          数据画像
    task_inference.py     规则任务推断
    eda.py                EDA 图表生成
    ml.py                 sklearn 建模流程
  web/
    index.html            网页端
    styles.css            页面样式
    app.js                页面交互逻辑
```

## 输出文件

运行任务后，默认会在 `storage/` 下生成：

```text
storage/
  datasets/     上传的原始数据
  processed/    清洗后的数据、特征预览、预处理摘要、预测结果
  artifacts/    EDA 图表、评估图表、特征重要性图
  models/       训练好的模型文件
  reports/      Markdown 报告和 context.json
```

其中 `processed/<job_id>/` 会包含：

- `cleaned.csv`：去重、删除目标列缺失样本、日期特征展开、缺失值填补后的训练数据
- `feature_preview.csv`：缺失值填补后的训练特征预览
- `preprocessing_summary.json`：预处理摘要，包括删除行数、字段类型、每列缺失值填补数量和值、编码策略等
- `predictions_*.csv`：新数据预测结果

评估图表包括：

- 分类任务：`confusion_matrix.png`
- 回归任务：`predicted_vs_actual.png`、`residual_plot.png`

特征重要性包括：

- `feature_importance.png`
- `feature_importance.csv`

`context.json` 会保存本次分析的完整结构化上下文，适合后续做 trace、评测和多轮追问。

## 项目定位

这个项目适合作为算法工程师实习项目的第一阶段：

- 展示基础 ML Pipeline 能力
- 展示 Agent 与传统算法流程结合的能力
- 展示 FastAPI 工程化能力
- 展示数据分析、建模、评估和报告生成闭环
- 展示对可控 Agent 边界的理解

后续可扩展方向：

- SHAP 可解释性分析
- 自动超参数搜索
- 多轮追问
- 分析任务 trace 可视化
- 多 Agent 协作
- 更多公开数据集 benchmark
- 数据清洗策略人工确认
- 代码仓库理解与自动修复 Agent
