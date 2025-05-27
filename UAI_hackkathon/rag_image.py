
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import os
import time
import numpy as np
import json

# ====== CẤU HÌNH ======

# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# sample_text = "Xin chào, tôi muốn thử embedding"

# embedding_vector = embedding_model.encode(sample_text)

# print(embedding_vector[:5])

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:7b"
OPENAI_MODEL = "gpt-4o"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-QADlU4SXVcLnIFUnH7IZVQ")
imageURL = "https://heramo.com/blog/wp-content/uploads/2024/01/cach-sua-tui-xach-bi-troc-da-1.jpg"

def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
client = OpenAI(
    base_url="https://llm.chotot.org",
    api_key=OPENAI_API_KEY
)

# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {
#             "role": "system",
#             "content": (
#                 "Bạn là chuyên gia đánh giá giá trị vật dụng dựa trên hình ảnh. "
#                 "Cho biết: 1. Vật thể trong ảnh là gì? 2. Mô tả vật thể một cách ngắn gọn. 3. Mức giá trung bình của vật thể này tại Việt Nam (bằng VND)."
#                 "Trả lời dưới dạng JSON với cấu trúc: "
#                 "[{'name': <tên vật>, 'description': <mô tả>, 'price_low': <giá thấp>, 'price_high': <giá cao>}]."
#             )
#         },
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Trong hình này có vật gì và giá của nó khoảng bao nhiêu tiền?"},
#                 {"type": "image_url", "image_url": {"url": imageURL}}
#             ]
#         }
        
#     ]
# )

def strip_code_block(text: str) -> str:
    """
    Nếu AI trả về JSON bên trong khối markdown (```json ... ```), loại bỏ các ký hiệu đó.
    """
    if text.startswith("```json") or text.startswith("```"):
        return text.strip("`").split("\n", 1)[1].rsplit("\n", 1)[0]
    return text

def call_openai() -> list:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia đánh giá giá trị vật dụng dựa trên hình ảnh. "
                        "Cho biết: 1. Vật thể trong ảnh là gì? 2. Mô tả vật thể một cách ngắn gọn. "
                        "3. Mức giá trung bình của vật thể này tại Việt Nam (bằng VND). "
                        "4. Độ phần trăm còn mới của vật thể (0-100%). "
                         "Trả lời dưới dạng JSON với cấu trúc: "
                        "[{'name': <tên vật>, 'description': <mô tả>, 'price_low': <giá thấp>, 'price_high': <giá cao>, 'percent_new': <độ phần trăm còn mới>}]. "
                         "Chỉ trả kết quả đúng theo JSON format, không thêm bất kỳ mô tả, tiêu đề hay lời dẫn nào."
                )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Trong hình này có vật gì và giá của nó khoảng bao nhiêu tiền?"},
                        {"type": "image_url", "image_url": {"url": imageURL}}
                    ]
                }
            ]
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

def call_ollama() -> str:
    try:
        # Tạo prompt tương tự nội dung system + user trong call_openai
        prompt = (
            "Bạn là chuyên gia đánh giá giá trị vật dụng dựa trên hình ảnh. "
            "Cho biết: 1. Vật thể trong ảnh là gì? 2. Mô tả vật thể một cách ngắn gọn. "
            "3. Mức giá trung bình của vật thể này tại Việt Nam (bằng VND). "
            "Trả lời dưới dạng JSON với cấu trúc: "
            "[{'name': <tên vật>, 'description': <mô tả>, 'price_low': <giá thấp>, 'price_high': <giá cao>}].\n"
            f"Hình ảnh: {imageURL}\n"
            "Trong hình này có vật gì và giá của nó khoảng bao nhiêu tiền?"
        )

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        result = response.json().get("response", "").strip()
        return result

    except requests.RequestException as e:
        return f"[Lỗi khi gọi Ollama] {e}"

# print(response.choices[0].message.content)

if __name__ == "__main__":
    # prompt = build_prompt(CUSTOMER_EATING_DATA)
    # print("=== Prompt gửi đến mô hình ===\n", prompt)
    print("\n=== Bắt đầu phân tích từ AI ===\n")
    print(call_openai())
    print("\n=== Kết thúc ===")