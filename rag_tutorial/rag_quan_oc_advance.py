from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import requests
import json
from datetime import datetime

app = FastAPI()

# -----------------------------
# Config & Data Schema
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

class CustomerContext(BaseModel):
    customer_id: str
    current_dishes: str
    spice_preference: str
    has_allergy: str
    is_first_time: str
    weather: str
    is_drinking: str
    group_size: int
    previous_favorite: Optional[str] = ""

# -----------------------------
# Static Menu
# -----------------------------
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

# -----------------------------
# Prompt Builder
# -----------------------------
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

Gợi ý 2–3 món ốc hoặc hải sản phù hợp tiếp theo, mỗi món nên kèm lý do ngắn gọn.
Nếu khách đã gọi món hấp hoặc luộc, nên gợi ý món chiên hoặc nướng để cân bằng khẩu vị.
Nếu nhóm từ 3 người trở lên và có uống bia, nên ưu tiên món nhậu dễ chia phần.

Trả lời dưới dạng danh sách:
1. [Tên món] – [Lý do]
2. [Tên món] – [Lý do]
...

Trả lời ngắn gọn, gần gũi, như một chủ quán thân thiện đang tư vấn tại bàn.
"""

def enrich_context(data: CustomerContext) -> str:
    hints = []
    if "hấp" in data.current_dishes or "luộc" in data.current_dishes:
        hints.append("Khách đã gọi món hấp/luộc, nên gợi ý món chiên hoặc nướng để cân bằng.")
    if data.group_size >= 3 and "bia" in data.is_drinking.lower():
        hints.append("Khách đi nhóm đông và có bia, nên gợi ý món nhậu dễ chia phần.")
    return "\n".join(hints)

def build_prompt(data: CustomerContext) -> str:
    hint_text = enrich_context(data)
    filled_prompt = PROMPT_TEMPLATE.format(**data.dict(), menu=MENU)
    return f"{hint_text}\n\n{filled_prompt}" if hint_text else filled_prompt

# -----------------------------
# Ollama Client
# -----------------------------
def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as e:
        return f"[Lỗi khi gọi Ollama] {e}"

if __name__ == "__main__":
    prompt = build_prompt(CUSTOMER_EATING_DATA)
    print("=== Prompt gửi đến mô hình ===\n", prompt)
    print("\n=== Gợi ý món tiếp theo từ AI ===\n")
    print(call_ollama(prompt))
    print("\n=== Kết thúc ===")
