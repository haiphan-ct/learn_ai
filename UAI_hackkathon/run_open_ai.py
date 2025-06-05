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

def build_messages(base64_images: list[str]) -> list:
    return [
        {
            "role": "system",
            "content": (
                "Bạn là chuyên gia đánh giá giá trị vật dụng dựa trên hình ảnh.\n"
                "Nhiệm vụ của bạn:\n"
                "- **Chỉ phân tích một vật thể duy nhất.**\n"
                "- Nếu hình ảnh bao gồm **nhiều vật thể khác nhau**, hãy trả về cảnh báo yêu cầu người dùng chụp lại hình rõ ràng, mỗi vật thể chụp riêng.\n"
                "- Nếu nhận diện được dòng sản phẩm cụ thể (ví dụ: iPhone 7 Plus, Galaxy S10), hãy ghi rõ đầy đủ tên model vào trường \"name\".\n"
                "Với vật thể đó, hãy cung cấp các thông tin sau:\n"
                "1. Tên vật thể (name)\n"
                "2. Mô tả vật thể ngắn gọn (description)\n"
                "3. Mức độ còn mới 0-100% (percent_new)\n"
                "4. Mô tả các hư hỏng nếu có (damages)\n"
                "5. Tỷ lệ hư hỏng 0-100% (damages_percent)\n"
                "6. Giá trung bình tại Việt Nam (price_low - price_high)\n"
                "7. Danh mục phù hợp (category) theo đúng cấu trúc sau:\n"
                "   category: {\n"
                "       \"level_1\": \"\",\n"
                "       \"level_2\": \"\",\n"
                "       \"level_3\": \"\",\n"
                "       \"level_4\": \"\",\n"
                "       \"level_5\": \"\"\n"
                "   }\n"
                "8. Cho biết những ảnh nào thuộc vật thể đó (image_indexes: [0, 1, 2]...)\n"
                "9. Liệt kê các tình trạng cụ thể ảnh hưởng đến giá trị (issues: [\"trầy\", \"móp\"]...)\n"
                "Chỉ chọn từ các danh mục có trong danh sách sau:\n"
                f"{category_examples}\n"
                "Nếu có hình ảnh **quá mờ hoặc không rõ vật thể**, hãy làm như sau:\n"
                "- Không cố đoán nếu không chắc chắn.\n"
                "- Trả về \"name\": \"Không rõ\", \"description\": \"Không thể nhận diện do ảnh mờ\"\n"
                "- Đặt tất cả các giá trị còn lại như \"percent_new\", \"price_low\", \"category\"… thành null hoặc 0.\n"
                "- Cảnh báo rõ ràng trong trường \"description\".\n"
                "- Chỉ rõ ảnh nào bị mờ bằng trường \"image_indexes\".\n"
                "Nếu phát hiện nhiều vật thể, hãy trả về JSON như sau:\n"
                "{\n"
                "  \"error\": \"Phát hiện nhiều vật thể khác nhau trong hình. Vui lòng chụp mỗi vật thể một cách riêng biệt và rõ ràng.\"\n"
                "}\n"
                "Trả kết quả đúng định dạng JSON như sau (nếu chỉ có một vật thể):\n"
                "[{\n"
                "  \"name\": \"\",\n"
                "  \"description\": \"\",\n"
                "  \"condition\": \"\",\n"
                "  \"damages\": \"\",\n"
                "  \"damages_percent\": 0,\n"
                "  \"percent_new\": 0,\n"
                "  \"price_low\": 0,\n"
                "  \"price_high\": 0,\n"
                "  \"issues\": [],\n"
                "  \"category\": {\n"
                "       \"level_1\": {\"id\": \"level_1_id\", \"name\": \"\"},\n"
                "       \"level_2\": {\"id\": \"level_2_id\", \"name\": \"\"},\n"
                "       \"level_3\": {\"id\": \"level_3_id\", \"name\": \"\"},\n"
                "       \"level_4\": {\"id\": \"level_4_id\", \"name\": \"\"},\n"
                "       \"level_5\": {\"id\": \"level_5_id\", \"name\": \"\"}\n"
                "   },\n"
                "  \"image_indexes\": [0, 1]\n"
                "}]\n"
            )
        }
    ]

if __name__ == "__main__":
    print("\n=== Bắt đầu phân tích từ AI ===\n")
    result = call_with_retry(base64_images)
    print("\n🟢 Kết quả phân tích:\n", result)
    print("\n=== Kết thúc ===")