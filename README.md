# 🏭 工业时间序列数据分析智能体 (Industrial AI Data Scientist Agent)

**SRTP（大学生创新创业训练计划）项目** 
> 致力于将大语言模型（LLM Agent）引入工业数据处理领域，打造一个“能看懂CSV、会自己选算法、全自动出报告”的本地化 AI 架构。

---

## 📂 项目架构 (Project Architecture)

当前项目采用模块化设计，清晰地分离了“大脑”、“工具”和“数据”。

```text
📦 srtp
 ┣ 📂 agent/                   # 🧠 智能中枢 (LLM 大脑模块)
 ┃ ┣ 📜 core_brain.py          # 核心调度器：负责接收指令、规划任务、调用工具
 ┃ ┣ 📜 prompts.py             # 提示词库：为 AI 设定人设与工作流标准
 ┃ ┗ 📜 __init__.py
 ┣ 📂 tools/                   # 🔨 算法工具箱 (Function Calling 工具库)
 ┃ ┣ 📜 data_imputation.py     # 缺失值补全：多项式插值、均值填充等
 ┃ ┣ 📜 anomaly_detection.py   # 异常检测：3-Sigma, 孤立森林等（待接入GitHub开源库）
 ┃ ┣ 📜 time_series_forecast.py# 时序预测：ARIMA, Prophet等
 ┃ ┗ 📜 __init__.py
 ┣ 📂 data/                    # 🧪 工业数据存放区 (由 MCP 机制提供数据感知)
 ┃ ┗ 📜 sample_data.csv        # 测试用时序数据集
 ┣ 📜 main.py                  # 🚀 系统统一启动入口 (目前已跑通 Tool Call 雏形)
 ┣ 📜 requirements.txt         # ⚙️ 环境依赖表 (Pandas, OpenAI, 等)
 ┣ 📜 .env                     # 🔒 密钥保管箱 (被 .gitignore 保护)
 ┗ 📜 .gitignore               # 🛡️ Git 上传白名单配置
```

---

## 📈 当前已完成进度 (Current Progress)

1. **环境与基建搭建**：
   - 完成 Python 与 Git 的标准化工业开发环境搭建。
   - 配置环境隔离机制与凭证保护机制（.env）。
2. **底层通讯与链路打通**：
   - 成功接入国内头部大语言模型（DeepSeek）API。
   - 解决 Windows 系统下底层文本隐形字符等通讯劫持 Bug。
3. **Agent 核心范式验证**：
   - 成功在本地以 main.py 跑通标准的 **Function Calling** 机制。
   - 证实了 LLM 可以根据自然语言自主决策并提取参数，驱动本地 Python 函数（工具）的执行闭环。
4. **全自动闭环交互与算法引入（最新升级！）**：
   - 构建了“数据感知”层：大模型可使用 `Pandas` 自动读取并总结 `data` 文件夹表特征。
   - 实现了孤立森林异常检测、均值数据插补算法的 `tools` 封装。
   - 打通了多轮通讯：现在 `main.py` 能够自主编排子任务，实现“读数据 -> 补数据 -> 找异常”的全自动流水线。

## 🎯 下一步计划 (Next Steps)

1. **接入高级预测算法**：完善 `time_series_forecast.py` 中的 ARIMA/Prophet 真实依赖封装。
2. **数据可视化与前端展示**：通过大模型直接生成代码并输出 Echarts 或 Matplotlib 图表，构建简易的网页 UI。
3. **Agent 工作流进阶**：从单一 Agent 升级为多智能体协同（例如一个做数据处理，一个专门负责校验）。

