
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import os
import time
import numpy as np

# ====== CẤU HÌNH ======

# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# sample_text = "Xin chào, tôi muốn thử embedding"

# embedding_vector = embedding_model.encode(sample_text)

# print(embedding_vector[:5])

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:7b"
OPENAI_MODEL = "gpt-4o"
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-QADlU4SXVcLnIFUnH7IZVQ")
if not OPENAI_API_KEY:
    raise EnvironmentError("Bạn cần đặt biến môi trường OPENAI_API_KEY trước khi chạy.")

client = OpenAI(
    base_url="https://llm.chotot.org",
    api_key=OPENAI_API_KEY
)

# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# print(embedding_model.data[0].embedding[:5])

# ====== Dữa liệu ======
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
    {"name": "Mô hình Pikachu đứng vẩy", "category": "Đồ chơi hoạt hình", "owner_note": "Mới 99%, thích đổi đồ cute"},
    {"name": "Sticker Doraemon vintage", "category": "Vật phẩm sưu tầm", "owner_note": "Tặng kèm nếu ai có mô hình thú"},
    {"name": "Baby Tree nhân vật số 07", "category": "Đồ chơi sưu tầm", "owner_note": "Trùng mẫu, muốn đổi mẫu khác"},
    {"name": "Móc khóa hình Gấu Brown", "category": "Phụ kiện mini", "owner_note": "Tìm người cùng gu đổi đồ"}
]

# ====== EMBEDDING & FILTER ======

# def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list:
#     response = client.embeddings.create(
#         model=model,
#         input=[text]
#     )
#     return response.data[0].embedding

def get_local_embedding(text: str) -> list:
    return EMBEDDING_MODEL.encode(text).tolist()

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def filter_items_by_embedding(user_text, items, top_k=3):
    user_vec = get_local_embedding(user_text)
    scored_items = []

    for item in items:
        if item['name'] == USER_ITEM_DATA['item_name']:
            continue  # skip trùng tên
        item_text = f"{item['name']} - {item['category']}. {item['owner_note']}"
        item_vec = get_local_embedding(item_text)
        score = cosine_similarity(user_vec, item_vec)
        scored_items.append((item, score))

    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored_items[:top_k]]

# ====== PROMPT ======

def format_exchange_history(history_list):
    return "\n".join(f"  + {item}" for item in history_list)

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
- Nếu nhu cầu gấp, có thể đề xuất dùng \"đẩy tin nổi bật\" để tăng lượt xem.
"""

def build_prompt(user_data, all_items):
    filtered_items = filter_items_by_embedding(
        user_data["exchange_wish"],
        all_items,
        top_k=3
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

# ====== CALL OPENAI ======

def call_openai(prompt: str, model: str = OPENAI_MODEL, max_retry: int = 3) -> str:
    for attempt in range(1, max_retry+1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Bạn là một hệ thống chuyển đổi đồ cho khách hàng."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
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

# def call_ollama() -> str:
#     try:
#         # Tạo prompt tương tự nội dung system + user trong call_openai
#         prompt = (
#             "Bạn là chuyên gia đánh giá giá trị vật dụng dựa trên hình ảnh. "
#             "Cho biết: 1. Vật thể trong ảnh là gì? 2. Mô tả vật thể một cách ngắn gọn. "
#             "3. Mức giá trung bình của vật thể này tại Việt Nam (bằng VND). "
#             "Trả lời dưới dạng JSON với cấu trúc: "
#             "[{'name': <tên vật>, 'description': <mô tả>, 'price_low': <giá thấp>, 'price_high': <giá cao>}].\n"
#             f"Hình ảnh: {imageURL}\n"
#             "Trong hình này có vật gì và giá của nó khoảng bao nhiêu tiền?"
#         )

#         payload = {
#             "model": OLLAMA_MODEL,
#             "prompt": prompt,
#             "stream": False
#         }

#         response = requests.post(OLLAMA_URL, json=payload)
#         response.raise_for_status()

#         result = response.json().get("response", "").strip()
#         return result

#     except requests.RequestException as e:
#         return f"[Lỗi khi gọi Ollama] {e}"

# print(response.choices[0].message.content)