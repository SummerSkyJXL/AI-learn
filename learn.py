import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载同目录下的 .env 文件
load_dotenv()

# 2. 从环境变量中获取 Key
# 这样代码里就不会出现明文的 Key 了
api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

try:
    print("正在通过安全配置连接 DeepSeek...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "你好！我已经学会了如何安全地管理 API Key。"}
        ]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"出错啦: {e}")

print("简单的更改用于测试git")