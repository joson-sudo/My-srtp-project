import os
import json
import pandas as pd
from openai import OpenAI

# 从 tools 导入我们刚刚写的模块
import sys
sys.path.append(os.path.dirname(__file__))
from tools.anomaly_detection import detect_anomalies
from tools.data_imputation import impute_missing_values

# 1. 安全读取配置
api_key = None
env_path = os.path.join(os.path.dirname(__file__) or ".", ".env")
try:
    with open(env_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if "=" in line and "DEEPSEEK" in line:
                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
except Exception: pass

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def extract_data_summary(file_path: str = "data/sample_data.csv"):
    """读取本地CSV概况"""
    print(f"\n[后台系统]：🔧 正在用 Pandas 分析 {file_path}...")
    try:
        df = pd.read_csv(file_path)
        summary = f"数据集总共有 {len(df)} 行。\n包含列：{list(df.columns)}。\n各列缺失值的数量为：\n{df.isnull().sum().to_string()}\n各列的数值统计信息为：\n{df.describe().to_string()}"
        return json.dumps({"status": "success", "data_summary": summary}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

# =====================================================================
# ★ 工具说明书库 (包含感知层、填补、异常检测)
# =====================================================================
tools_description = [
    {
        "type": "function",
        "function": {
            "name": "extract_data_summary",
            "description": "第一步调用：读取传感器数据表的情况（有无空缺、最大最小值等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径，默认 'data/sample_data.csv'"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "impute_missing_values",
            "description": "第二步调用：用来填补表格中特定列(比如 temperature)的缺失值空缺",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                    "column": {"type": "string", "description": "需要填补缺失值的列名，例如 'temperature'"},
                    "method": {"type": "string", "enum": ["mean", "forward"], "description": "填补方法"}
                },
                "required": ["file_path", "column", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "第三步调用：用来检测某一列中的异常值(如温度突然超高)",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "column": {"type": "string", "description": "要检测的列"}
                },
                "required": ["file_path", "column"]
            }
        }
    }
]

print("📡 开始多轮对话：让大模型完成闭环任务：\n")
user_prompt = "大牛，我们的任务是自动诊断 data/sample_data.csv。\n请按顺序执行：1. 先读取数据概况；2. 如果有缺失值，请补全temperature列；3. 补全后立刻使用孤立森林检测该列的异常值。每做完一步请向我汇报！"
messages = [{"role": "user", "content": user_prompt}]

# 由于需要大模型按顺序调用三个工具，我们写一个循环让他可以“思考并行动”3次
for step in range(4):
    print(f"--- 循环思考 第 {step+1} 轮 ---")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools_description,
        tool_choice="auto"
    )
    
    ai_msg = response.choices[0].message
    messages.append(ai_msg) # 把大模型的对话加入记忆
    
    if getattr(ai_msg, "tool_calls", None):
        tool_call = ai_msg.tool_calls[0]
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"🤖 [大模型决策]：需要调用工具 => {func_name}，参数：{args}")
        
        # 路由到对应的本地函数
        if func_name == "extract_data_summary":
            res = extract_data_summary(**args)
        elif func_name == "impute_missing_values":
            res = impute_missing_values(**args)
        elif func_name == "detect_anomalies":
            res = detect_anomalies(**args)
        else:
            res = json.dumps({"error": "找不到工具"})
            
        print(f"✅ [本地执行结果]：{res}")
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res})
        
    else:
        print(f"\n✨ [大模型最终总结]：\n{ai_msg.content}")
        break  # 如果他不调工具了，说明任务完成了，退出循环


