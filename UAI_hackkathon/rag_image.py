
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import os
import time
import numpy as np
import json
import base64

# ====== CẤU HÌNH ======

# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# sample_text = "Xin chào, tôi muốn thử embedding"

# embedding_vector = embedding_model.encode(sample_text)

# print(embedding_vector[:5])

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:7b"
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-QADlU4SXVcLnIFUnH7IZVQ")
imageURL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTn537ZjqmyaZfVqdLJe4JIbhkYBkXMj5CQmQ&s"

def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
client = OpenAI(
    base_url="https://llm.chotot.org",
    api_key=OPENAI_API_KEY
)

image_path = "image_check.jpg"
base64_image = encode_image_to_base64(image_path)

def strip_code_block(text: str) -> str:
    """
    Nếu AI trả về JSON bên trong khối markdown (```json ... ```), loại bỏ các ký hiệu đó.
    """
    if text.startswith("```json") or text.startswith("```"):
        return text.strip("`").split("\n", 1)[1].rsplit("\n", 1)[0]
    return text

def build_messages(base64_image: str) -> list:
    return [
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia đánh giá giá trị vật dụng dựa trên hình ảnh. "
                        "Cho biết: "
                        "1. Tên vật thể (name) \n"
                        "2. Mô tả vật thể ngắn gọn (description) \n"
                        "3. Mức độ còn mới 0-100% (percent_new) \n"
                        "4. Mô tả các hư hỏng nếu có (damages) \n"
                        "5. Tỷ lệ hư hỏng 0-100% (damages_percent) \n"
                        "6. Giá trung bình tại Việt Nam (price_low - price_high) \n"
                        "Trả lời đúng định dạng JSON sau (chỉ trả JSON, không thêm mô tả):\n"
                        "[{"
                        "\"name\": \"\", "
                        "\"description\": \"\", "
                        "\"condition\": \"\", "
                        "\"damages\": \"\", "
                        "\"damages_percent\": 0, "
                        "\"percent_new\": 0, "
                        "\"price_low\": 0, "
                        "\"price_high\": 0, "
                         "Chỉ trả kết quả đúng theo JSON format, không thêm bất kỳ mô tả, tiêu đề hay lời dẫn nào."
                )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Trong hình này có vật gì và giá của nó khoảng bao nhiêu tiền?"},
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]

def call_openai(base64image: str) -> list:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=build_messages(base64image),
            temperature=0.5,
        )
        raw_content = response.choices[0].message.content.strip()
        print("✅ Kết quả trả về từ AI:\n", raw_content)

        json_str = strip_code_block(raw_content)

        if not json_str.startswith("["):
            print("❌ Dữ liệu không ở dạng JSON sau khi làm sạch. AI có thể đã trả lời bằng đoạn văn.")
            return []

        return json.loads(json_str)

    except json.JSONDecodeError as je:
        print("❌ Lỗi khi parse JSON từ OpenAI:", je)
        print("Nội dung AI trả về:\n", raw_content)
        return []

    except Exception as e:
        print("❌ Lỗi khi gọi OpenAI:", e)
        return []
    
def call_with_retry(base64image: str, retries=3, delay=2) -> list:
    """
    Gọi OpenAI với retry nếu gặp lỗi. Mặc định thử 3 lần, cách nhau 2 giây.
    """
    for attempt in range(1, retries + 1):
        print(f"🟡 Thử gọi OpenAI lần {attempt}")
        result = call_openai(base64image)
        if result:
            return result
        time.sleep(delay)
    print("❌ Hết số lần thử. Không thể lấy dữ liệu từ AI.")
    return []

if __name__ == "__main__":
    # prompt = build_prompt(CUSTOMER_EATING_DATA)
    # print("=== Prompt gửi đến mô hình ===\n", prompt)
    print("\n=== Bắt đầu phân tích từ AI ===\n")
    print(call_with_retry(base64_image))
    print("\n=== Kết thúc ===")