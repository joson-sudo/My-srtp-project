import os
import json
from openai import OpenAI

# 1. 安全读取配置（保持之前跑通的代码不变）
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

# =====================================================================
# ★ 新增内容：这是我们写的一个“本地工具” (未来的工业算法就会放在这种函数里)
# =====================================================================
def get_weather(location: str):
    """一个假的本地天气工具，其实是我们自己写的判断逻辑"""
    print(f"\n[后台系统]：🔧 探测到大模型想要调用工具，正在本地执行 get_weather(location='{location}')...")
    if "北京" in location:
        return json.dumps({"location": "北京", "temperature": "18度", "condition": "晴朗带一点霾"})
    elif "上海" in location:
        return json.dumps({"location": "上海", "temperature": "22度", "condition": "下雨"})
    else:
        return json.dumps({"location": location, "temperature": "未知", "condition": "我的工具里没录入这个城市"})

# =====================================================================
# ★ 我们要把工具的“说明书”写出来给大模型看
# =====================================================================
tools_description = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "当用户问到天气时，必须调用这个工具来获取准确温度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "用户提到的城市名字，例如 '北京'"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

print("📡 开始对话：大牛，请问北京现在天气怎么样？\n")

# =====================================================================
# ★ 正式沟通：把用户的提问和“工具箱”一起扔给大模型
# =====================================================================
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "大牛，请结合目前天气数据告诉我，北京现在天气怎么样？"}
    ],
    tools=tools_description,  # 划重点：告诉大模型“你手边有锤子可以用！”
    tool_choice="auto"        # 划重点：让大模型“自己决定”要不要用锤子
)

# =====================================================================
# ★ 见证奇迹：分析大模型的返回值大模型想要干啥？
# =====================================================================
ai_message = response.choices[0].message

if ai_message.tool_calls:
    # 说明大模型觉得自己的脑子算不出来，它决定“动用工具”了！
    print("🤖 [大模型]：我不知道天气，但我决定调用你给我的工具！")
    tool_call = ai_message.tool_calls[0]
    
    # 提取大模型从我们自然语言中“猜”出来的参数（比如“北京”）
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"🤖 [大模型请求]：请在本地帮我运行 {function_name} 函数，参数是：{arguments}")
    
    # 我们在本地帮它运行！
    if function_name == "get_weather":
        final_result = get_weather(location=arguments.get("location"))
        print(f"✅ [本地执行完毕]，得到的数据是：{final_result}")
else:
    # 大模型没用工具，纯靠自己脑补说话
    print(f"🤖 大模型直接回复了：{ai_message.content}")

