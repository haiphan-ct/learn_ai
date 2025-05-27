import requests
from openai import OpenAI
import os
import time

# ====== CẤU HÌNH ======

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:7b"
OPENAI_MODEL = "gpt-4o"

# Bắt buộc phải có API key trong biến môi trường
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-QADlU4SXVcLnIFUnH7IZVQ")
if not OPENAI_API_KEY:
    raise EnvironmentError("Bạn cần đặt biến môi trường OPENAI_API_KEY trước khi chạy.")

# ====== DỮ LIỆU NGƯỜI DÙNG & MÓN HIỆN CÓ ======

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

# ====== XỬ LÝ DỮ LIỆU & XÂY DỰNG PROMPT ======

def format_exchange_history(history_list):
    # Chuẩn hóa lịch sử trao đổi dạng bullet
    return "\n".join(f"  + {item}" for item in history_list)

def filter_available_items(items, exclude_name, exchange_wish):
    """
    Lọc món phù hợp:
    - Loại bỏ món trùng tên với món người dùng có
    - Ưu tiên món có từ khóa liên quan trong exchange_wish hoặc cùng category
    """
    # Lấy từ khóa chính trong exchange_wish (đơn giản lấy các từ có độ dài > 3)
    keywords = [w.lower() for w in exchange_wish.replace(",", "").split() if len(w) > 3]

    filtered = []
    for item in items:
        name_lower = item['name'].lower()
        category_lower = item['category'].lower()
        if item['name'] == exclude_name:
            continue  # Loại món trùng tên
        # Nếu món có từ khóa trùng trong tên hoặc category thì ưu tiên thêm
        matched_keyword = any(kw in name_lower or kw in category_lower for kw in keywords)
        if matched_keyword:
            filtered.append(item)

    # Nếu không có món nào khớp keyword, fallback lấy món cùng category
    if not filtered:
        filtered = [item for item in items if item['name'] != exclude_name and item['category'] == USER_ITEM_DATA['category']]

    # Nếu vẫn không có thì trả về tất cả món trừ món trùng tên
    if not filtered:
        filtered = [item for item in items if item['name'] != exclude_name]

    return filtered

def format_available_items(items):
    return "\n".join(
        f"- {item['name']} ({item['category']}): {item['owner_note']}"
        for item in items
    )

PROMPT_TEMPLATE = """
Bạn là trợ lý thông minh giúp kết nối người có đồ muốn trao đổi với người có món phù hợp.

Người dùng có:
- Tên món: {item_name}
- Danh mục: {category}
- Tình trạng: {condition}
- Mong muốn đổi lấy: {exchange_wish}
- Mức độ khẩn cấp: {urgency}
- Lịch sử trao đổi:
{exchange_history}

Danh sách món đồ hiện có trong hệ thống:
{available_items}

Yêu cầu:
- Gợi ý 2–3 món phù hợp với nhu cầu người dùng.
- Mỗi gợi ý nên nêu tên món + lý do gợi ý ngắn gọn.
- Tránh gợi lại món người dùng đang có hoặc không phù hợp.
- Nếu nhu cầu gấp, có thể đề xuất dùng "đẩy tin nổi bật" để tăng lượt xem.

Ví dụ:
1. Baby Tree nhân vật số 03 – Cùng dòng Baby Tree, khác mẫu, phù hợp nhu cầu bạn.
2. Mô hình nhỏ thú hoạt hình Pikachu – Món cute, gần như mới, phù hợp trao đổi.
3. Móc khóa hình Gấu Brown – Phù hợp gu bạn, tiện đổi thêm phụ kiện mini.

Trả lời ngắn gọn, gần gũi như đang tư vấn trong hội nhóm đổi đồ.
"""

def build_prompt(user_data, all_items):
    filtered_items = filter_available_items(
        all_items,
        exclude_name=user_data["item_name"],
        exchange_wish=user_data["exchange_wish"]
    )
    return PROMPT_TEMPLATE.format(
        item_name=user_data["item_name"],
        category=user_data["category"],
        condition=user_data["condition"],
        exchange_wish=user_data["exchange_wish"],
        urgency=user_data["urgency"],
        exchange_history=format_exchange_history(user_data["exchange_history"]),
        available_items=format_available_items(filtered_items)
    )

# ====== GỌI API OPENAI với retry và xử lý lỗi ======

client = OpenAI(
    base_url="https://llm.chotot.org",
    api_key=OPENAI_API_KEY
)

def call_openai(prompt: str, model: str = OPENAI_MODEL, max_retry: int = 3) -> str:
    for attempt in range(1, max_retry+1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Bạn là một hệ thống chuyển đổi đồ cho khách hàng."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5  # Giảm nhiệt độ để output ổn định hơn
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Lỗi khi gọi OpenAI (lần {attempt}): {e}")
            if attempt < max_retry:
                print("Đang thử lại sau 2 giây...")
                time.sleep(2)
            else:
                return f"Lỗi khi gọi OpenAI sau {max_retry} lần: {e}"

# ====== MAIN ======

if __name__ == "__main__":
    prompt = build_prompt(USER_ITEM_DATA, AVAILABLE_ITEMS)
    print("=== Prompt gửi đến mô hình ===\n")
    print(prompt)
    print("\n=== Gợi ý từ AI ===\n")
    result = call_openai(prompt)
    print(result)
    print("\n=== Kết thúc ===")
