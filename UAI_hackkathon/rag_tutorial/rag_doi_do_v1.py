import requests

import openai
from openai import OpenAI
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:7b"
OPENAI_MODEL = "gpt-4o"

# Set your OpenAI API key (either directly or via environment variable)
# openai.api_key = os.getenv("OPENAI_API_KEY", "sk-QADlU4SXVcLnIFUnH7IZVQ")

USER_ITEM_DATA = {
    "user_id": "U10092",
    "item_name": "Baby Tree nhân vật số 07",
    "category": "Đồ chơi sưu tầm",
    "condition": "Đã khui hộp, còn mới",
    "exchange_wish": "Muốn đổi lấy nhân vật baby tree khác chưa có hoặc mô hình nhỏ thú hoạt hình",
    "urgency": "Cao",
    "exchange_history": [
        "Từng đổi sách thiếu nhi với người trong group Sách Cũ Chất",
        "Đổi mô hình Lego mini với bạn trong hội Sưu tầm mini figure"
    ]
}

AVAILABLE_ITEMS = [
    {"name": "Baby Tree nhân vật số 03", "category": "Đồ chơi sưu tầm", "owner_note": "Đổi món tương đương, ưu tiên baby tree khác"},
    {"name": "Mô hình Pikachu đứng vẫy", "category": "Đồ chơi hoạt hình", "owner_note": "Mới 99%, thích đổi đồ cute"},
    {"name": "Sticker Doraemon vintage", "category": "Vật phẩm sưu tầm", "owner_note": "Tặng kèm nếu ai có mô hình thú"},
    {"name": "Baby Tree nhân vật số 07", "category": "Đồ chơi sưu tầm", "owner_note": "Trùng mẫu, muốn đổi mẫu khác"},
    {"name": "Móc khóa hình Gấu Brown", "category": "Phụ kiện mini", "owner_note": "Tìm người cùng gu đổi đồ"}
]

# # Dữ liệu khách đang dùng món
# CUSTOMER_EATING_DATA = {
#     "customer_id": "KH56789",
#     "current_dishes": "Ốc lác xào tỏi, Ốc bươu hấp sả",
#     "spice_preference": "Cay vừa",
#     "has_allergy": "Không",
#     "is_first_time": "Không",
#     "weather": "Trời mát, buổi tối",
#     "is_drinking": "Có (bia)",
#     "group_size": 3,
#     "previous_favorite": "Nghêu hấp Thái, Sò huyết nướng"
# }

# MENU = """
# Menu quán gồm:
# - Ốc hương rang muối
# - Sò lông nướng mỡ hành
# - Nghêu hấp sả
# - Ốc tỏi cháy tỏi
# - Mực chiên nước mắm
# - Tôm nướng muối ớt
# - Sò điệp phô mai
# - Hàu nướng mỡ hành
# - Bạch tuộc nướng sa tế
# - Ốc len xào dừa
# """


# Prompt template
# PROMPT_TEMPLATE = """
# {menu}
# Bạn là một chủ quán ốc có kinh nghiệm, nhiệm vụ của bạn là gợi ý món tiếp theo cho khách đang dùng bữa.

# Thông tin khách hàng:
# - Mã khách: {customer_id}
# - Món đang ăn: {current_dishes}
# - Mức độ ăn cay: {spice_preference}
# - Dị ứng: {has_allergy}
# - Lần đầu đến quán: {is_first_time}
# - Thời tiết hiện tại: {weather}
# - Có uống bia: {is_drinking}
# - Số người trong nhóm: {group_size}
# - Món yêu thích trước đây: {previous_favorite}

# Yêu cầu:
# Dựa trên món đã ăn, sở thích, thời tiết và bối cảnh, bạn sẽ gợi ý **2–3 món ốc hoặc hải sản phù hợp tiếp theo**, mỗi món nên kèm **lý do ngắn gọn** để giúp khách dễ chọn.

# Ví dụ cách trả lời:
# 1. [Tên món] – [Lý do gợi ý]
# 2. ...
# 3. ...

# Trả lời ngắn gọn, gần gũi, như một chủ quán thân thiện đang tư vấn tại bàn.
# """

PROMPT_TEMPLATE = """
Bạn là trợ lý thông minh giúp kết nối người có đồ muốn trao đổi với người có món phù hợp.

Người dùng có:
- Tên món: {item_name}
- Danh mục: {category}
- Tình trạng: {condition}
- Mong muốn đổi lấy: {exchange_wish}
- Mức độ khẩn cấp: {urgency}
- Lịch sử trao đổi: {exchange_history}

Danh sách món đồ hiện có trong hệ thống:
{available_items}

Yêu cầu:
- Gợi ý 2–3 món phù hợp với nhu cầu người dùng.
- Mỗi gợi ý nên nêu tên món + lý do gợi ý ngắn gọn.
- Tránh gợi lại món người dùng đang có (ví dụ: đã trùng Baby Tree số 07).
- Nếu nhu cầu gấp, có thể thêm gợi ý sử dụng tính năng "đẩy tin nổi bật" để tăng lượt xem.

Trả lời gần gũi, ngắn gọn như đang tư vấn trong hội nhóm đổi đồ.
"""

# def build_prompt(data: dict) -> str:
#     return PROMPT_TEMPLATE.format(**data, menu=MENU)

# def build_prompt(data: dict) -> str:
#     return PROMPT_TEMPLATE.format(**data, available_items=AVAILABLE_ITEMS)

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

def format_available_items(items, exclude_name):
    return "\n".join(
        f"- {item['name']} ({item['category']}): {item['owner_note']}"
        for item in items
        if item['name'] != exclude_name
    )

def build_prompt(user_data, items):
    return PROMPT_TEMPLATE.format(
        item_name=user_data["item_name"],
        category=user_data["category"],
        condition=user_data["condition"],
        exchange_wish=user_data["exchange_wish"],
        urgency=user_data["urgency"],
        exchange_history="\n".join(f"  + {h}" for h in user_data["exchange_history"]),
        available_items=format_available_items(items, user_data["item_name"])
    )

# ========== 3. GỌI OPENAI ==========

# client = OpenAI(
#     base_url="https://llm.chotot.org/v1",
#     api_key=os.getenv("OPENAI_API_KEY", "sk-QADlU4SXVcLnIFUnH7IZVQ")
# )

client = OpenAI(
    base_url="https://llm.chotot.org",  # Đảm bảo đường dẫn đúng định dạng
    api_key=os.getenv("OPENAI_API_KEY", "sk-QADlU4SXVcLnIFUnH7IZVQ")  # API key nội bộ nếu cần
)

def call_openai(prompt: str, model: str = OPENAI_MODEL) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một hệ thống chuyển đổi đồ cho khách hàng."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi khi gọi OpenAI: {e}"

if __name__ == "__main__":
#     return PROMPT_TEMPLATE.format(**data, available_items=AVAILABLE_ITEMS)
    prompt = build_prompt(USER_ITEM_DATA, AVAILABLE_ITEMS)
    print("=== Prompt gửi đến mô hình ===\n", prompt)
    print("\n=== Gợi ý món tiếp theo từ AI ===\n")
    print(call_openai(prompt))
    print("\n=== Kết thúc ===")