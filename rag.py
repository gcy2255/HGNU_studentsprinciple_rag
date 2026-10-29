# api_key=sk-ciz821vynbsxrqbxbnfw99279h5093ra09vxp5q4a4mru49i
# url_openai=https://api.xiaomimimo.com/v1
MIMO_API_KEY="sk-ciz821vynbsxrqbxbnfw99279h5093ra09vxp5q4a4mru49i"
import os
from openai import OpenAI
import json

client = OpenAI(
    api_key=MIMO_API_KEY,
    base_url="https://api.xiaomimimo.com/v1"
)

completion = client.chat.completions.create(
    model="mimo-v2.6-pro",
    messages=[
        {
            "role": "system",
            "content": "You are MiMo, an AI assistant developed by Xiaomi. Today is date: Tuesday, December 16, 2025. Your knowledge cutoff date is December 2024."
        },
        {
            "role": "user",
            "content": "please introduce yourself"
        }
    ],
    max_completion_tokens=1024,
    temperature=1.0,
    top_p=0.95,
    stream=False,
    stop=None,
    frequency_penalty=0,
    presence_penalty=0
)

print(completion.model_dump_json())
data=completion.model_dump_json()
# content=data['content']