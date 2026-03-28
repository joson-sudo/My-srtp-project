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

## 🎯 下一步计划 (Next Steps)

1. **构建“数据感知”层**：引入 Pandas，让大模型能读取并总结 data 文件夹里的表格特征信息。
2. **封装备选算法库**：从 GitHub 开源代码中引入孤立森林与预测算法，装填进 tools 的各个 Python 文件中。
3. **闭环交互**：结合前两步，让大模型读完数据后自动调用对应算法并输出分析结果。

