import requests

import openai
from openai import OpenAI
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:7b"
OPENAI_MODEL = "gpt-4o"

# Set your OpenAI API key (either directly or via environment variable)
# openai.api_key = os.getenv("OPENAI_API_KEY", "sk-QADlU4SXVcLnIFUnH7IZVQ")

# Dữ liệu khách đang dùng món
CUSTOMER_EATING_DATA = {
    "customer_id": "KH56789",
    "current_dishes": "Ốc lác xào tỏi, Ốc bươu hấp sả",
    "spice_preference": "Cay vừa",
    "has_allergy": "Không",
    "is_first_time": "Không",
    "weather": "Trời mát, buổi tối",
    "is_drinking": "Có (bia)",
    "group_size": 3,
    "previous_favorite": "Nghêu hấp Thái, Sò huyết nướng"
}

MENU = """
Menu quán gồm:
- Ốc hương rang muối
- Sò lông nướng mỡ hành
- Nghêu hấp sả
- Ốc tỏi cháy tỏi
- Mực chiên nước mắm
- Tôm nướng muối ớt
- Sò điệp phô mai
- Hàu nướng mỡ hành
- Bạch tuộc nướng sa tế
- Ốc len xào dừa
"""


# Prompt template
PROMPT_TEMPLATE = """
{menu}
Bạn là một chủ quán ốc có kinh nghiệm, nhiệm vụ của bạn là gợi ý món tiếp theo cho khách đang dùng bữa.

Thông tin khách hàng:
- Mã khách: {customer_id}
- Món đang ăn: {current_dishes}
- Mức độ ăn cay: {spice_preference}
- Dị ứng: {has_allergy}
- Lần đầu đến quán: {is_first_time}
- Thời tiết hiện tại: {weather}
- Có uống bia: {is_drinking}
- Số người trong nhóm: {group_size}
- Món yêu thích trước đây: {previous_favorite}

Yêu cầu:
Dựa trên món đã ăn, sở thích, thời tiết và bối cảnh, bạn sẽ gợi ý **2–3 món ốc hoặc hải sản phù hợp tiếp theo**, mỗi món nên kèm **lý do ngắn gọn** để giúp khách dễ chọn.

Ví dụ cách trả lời:
1. [Tên món] – [Lý do gợi ý]
2. ...
3. ...

Trả lời ngắn gọn, gần gũi, như một chủ quán thân thiện đang tư vấn tại bàn.
"""

def build_prompt(data: dict) -> str:
    return PROMPT_TEMPLATE.format(**data, menu=MENU)

# def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
#     url = OLLAMA_URL
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False
#     }
#     try:
#         response = requests.post(url, json=payload)
#         response.raise_for_status()
#         data = response.json()
#         return data.get("response", "").strip()
#     except requests.exceptions.RequestException as e:
#         return f"Lỗi khi gọi Ollama: {e}"

client = OpenAI(
    base_url="https://llm.chotot.org/v1",  # Đảm bảo đường dẫn đúng định dạng
    api_key=os.getenv("OPENAI_API_KEY", "sk-QADlU4SXVcLnIFUnH7IZVQ")  # API key nội bộ nếu cần
)

def call_openai(prompt: str, model: str = OPENAI_MODEL) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một chủ quán ốc thân thiện và hiểu khách hàng."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi khi gọi OpenAI: {e}"

if __name__ == "__main__":
    prompt = build_prompt(CUSTOMER_EATING_DATA)
    print("=== Prompt gửi đến mô hình ===\n", prompt)
    print("\n=== Gợi ý món tiếp theo từ AI ===\n")
    print(call_openai(prompt))
    print("\n=== Kết thúc ===")