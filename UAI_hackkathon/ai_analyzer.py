import os
import time
import json
from openai import OpenAI
from keys import OPENAI_API_KEY_TEAM
from image_utils import load_category_examples_from_csv

OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY_TEAM)

client = OpenAI(
    base_url="https://llm.chotot.org",
    api_key=OPENAI_API_KEY
)

def strip_code_block(text: str) -> str:
    if text.startswith("```json") or text.startswith("```"):
        return text.strip("`").split("\n", 1)[1].rsplit("\n", 1)[0]
    return text

def build_messages(base64_images: list[str], category_examples: dict[int, str]) -> list:
    image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in base64_images
    ]
    return [
        {
            "role": "system",
            "content": (
                "Bạn là chuyên gia đánh giá giá trị vật dụng dựa trên hình ảnh.\n"
                "Nhiệm vụ của bạn:\n"
                "- **Chỉ phân tích duy nhất một vật thể trong tất cả ảnh.**\n"
                "- Trước khi phân tích, hãy kiểm tra **chất lượng từng ảnh**:\n"
                "  - Có bị mờ, thiếu sáng hoặc không rõ vật thể không?\n"
                "  - Góc nào trong ảnh (trái, phải, trên, dưới, trung tâm) không rõ?\n"
                "  - Nếu có góc không rõ, hãy **ghi chú rõ ràng** theo mẫu: \"Tấm hình thứ 1, góc trái không nhìn rõ, cần cải thiện hoặc chụp lại.\"\n"
                "- Nếu phát hiện **nhiều vật thể khác nhau**, hãy **không phân tích tiếp** và trả về JSON:\n"
                "  {\n"
                "    \"error\": \"Phát hiện nhiều vật thể khác nhau trong hình. Vui lòng chụp mỗi vật thể một cách riêng biệt và rõ ràng.\"\n"
                "  }\n"
                "- Nếu ảnh quá mờ hoặc không đủ rõ để định giá, không được đoán bừa. Trả về:\n"
                "  {\n"
                "    \"name\": \"Không rõ\",\n"
                "    \"description\": \"Không thể nhận diện do ảnh mờ hoặc không rõ vật thể. Tấm hình thứ 0 bị mờ góc phải.\",\n"
                "    \"condition\": null,\n"
                "    \"damages\": null,\n"
                "    \"damages_percent\": 0,\n"
                "    \"percent_new\": 0,\n"
                "    \"price_low\": 0,\n"
                "    \"price_high\": 0,\n"
                "    \"price_low_real\": 0,\n"
                "    \"price_high_real\": 0,\n"
                "    \"issues\": [],\n"
                "    \"category\": null,\n"
                "    \"image_indexes\": [0]\n"
                "  }\n"
                "- Nếu vật thể có các phụ kiện đi kèm như đồ chơi, nệm, hoặc tay cầm được gắn cố định hoặc là một phần của thiết kế (ví dụ như xe tập đi có đồ chơi phía trước), hãy coi toàn bộ đó là **một vật thể duy nhất** để phân tích. Không được tách riêng các phần này thành nhiều vật thể.\n"
                "- Nếu nhận diện được vật thể, hãy phân tích kỹ và cung cấp các thông tin sau:\n"
                "  1. Tên vật thể (name)\n"
                "  2. Mô tả vật thể ngắn gọn (description)\n"
                "  3. Mức độ còn mới 0-100% (percent_new)\n"
                "  4. Mô tả các hư hỏng nếu có (damages)\n"
                "  5. Tỷ lệ hư hỏng 0-100% (damages_percent)\n"
                "  6. Giá trung bình tại Việt Nam (price_low - price_high)\n"
                "  7. Giá trị thực tế theo mức độ còn mới:\n"
                "     - price_min = percent_new / 100 * price_low\n"
                "     - price_max = percent_new / 100 * price_high\n"
                "  8. Danh mục phù hợp theo định dạng:\n"
                "     \"category\": {\n"
                "         \"id\": \"\",\n"
                "         \"name\": \"\",\n"
                "     },\n"
                "  9. Liệt kê các ảnh liên quan đến vật thể đó (image_indexes: [0, 1, 2])\n"
                " 10. Liệt kê chi tiết các vấn đề ảnh hưởng giá trị (issues: [\"trầy\", \"móp\"]), càng chi tiết càng tốt, trầy/móp... ở đâu \n"
                " 11. Độ tin cậy của phân tích, giá trị từ 0.0 (không chắc chắn) đến 1.0 (rất chắc chắn):\n"
                "     \"confident\": 0.85\n"
                f"Chỉ chọn từ các danh mục có trong danh sách sau:\n{category_examples}\n"
                "Kết quả phải được trả về đúng định dạng JSON như ví dụ sau:\n"
                "Xoá field percent_new, price_low, price_high"
                "{\n"
                "  \"name\": \"iPhone 11\",\n"
                "  \"description\": \"Điện thoại Apple, màn hình 6.1 inch, màu đen. Ảnh thứ 1 góc trái hơi mờ.\",\n"
                "  \"condition\": \"Còn khá mới, có trầy nhẹ ở cạnh viền.\",\n"
                "  \"damages_percent\": 5,\n"
                "  \"price_min\": 4500000,\n"
                "  \"price_max\": 5500000,\n"
                "  \"issues\": [\"trầy nhẹ ở lưng điện thoại\", \"Móp ở viền\"],\n"
                "  \"category\": {\n"
                "    \"id\": 27,\n"
                "    \"name\": \"Điện thoại\"\n"
                "  },\n"
                "  \"image_indexes\": [0, 1],\n"
                "  \"image_issues\": [2, 3]\n"
                "  \"confident\": 0.85\n"
                "}"
            )
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Trong các hình này có vật gì và giá của chúng khoảng bao nhiêu tiền?"}] + image_messages
        }
    ]

def call_openai(base64_images: list[str], category_examples: dict[int, str]) -> list:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=build_messages(base64_images, category_examples),
            temperature=0.1,
        )
        raw_content = response.choices[0].message.content.strip()
        print("✅ Kết quả trả về từ AI:\n", raw_content)

        json_str = strip_code_block(raw_content)

        if not json_str.startswith("{"):
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

def call_with_retry(base64_images: list[str], category_examples: dict[int, str], retries=3, delay=2) -> list:
    for attempt in range(1, retries + 1):
        print(f"🟡 Thử gọi OpenAI lần {attempt}")
        result = call_openai(base64_images, category_examples)
        if result:
            return result
        time.sleep(delay)
    print("❌ Hết số lần thử. Không thể lấy dữ liệu từ AI.")
    return []

def build_messages_cate(name: str) -> list:
    category_examples = load_category_examples_from_csv("category.csv")
    return [
        {
            "role": "system",
            "content": (
                "Bạn là hệ thống phân loại danh mục cho vật dụng.\n"
                "Nhiệm vụ của bạn:\n"
                "- Nhận đầu vào là một tên sản phẩm hoặc mô tả (ví dụ: \"iphone 11\", \"máy giặt LG\").\n"
                "- Dựa vào nội dung đó, hãy **suy luận danh mục phù hợp nhất** trong danh sách bên dưới.\n"
                "- Trả về kết quả theo đúng định dạng JSON sau:\n"
                "{\n"
                "  \"category\": {\n"
                "    \"level_1\": {\"id\": \"\", \"name\": \"\"},\n"
                "    \"level_2\": {\"id\": \"\", \"name\": \"\"},\n"
                "    \"level_3\": {\"id\": \"\", \"name\": \"\"},\n"
                "    \"level_4\": {\"id\": \"\", \"name\": \"\"},\n"
                "    \"level_5\": {\"id\": \"\", \"name\": \"\"}\n"
                "  }\n"
                "}\n"
                "- Nếu không tìm được danh mục phù hợp, hãy trả về:\n"
                "{ \"error\": \"Không tìm thấy danh mục phù hợp cho đầu vào.\" }\n"
                "- Chỉ chọn từ các danh mục có trong danh sách sau:\n"
                f"{category_examples}\n"
            )
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Đầu vào: {name}\nHãy suy luận và trả về category phù hợp."}]
        }
    ]

def call_openai_detect_cate(name: str) -> list:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=build_messages_cate(name),
            temperature=0.1,
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