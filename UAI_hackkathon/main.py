from image_utils import encode_images_to_base64
from image_utils import get_image_paths_from_folder
from ai_analyzer import call_with_retry
from ai_analyzer import call_openai_detect_cate

category_examples = {
    1: "Quần áo bé gái",
    2: "Quần áo bé trai",
    3: "Giày dép bé gái",
    4: "Giày dép bé trai",
    5: "Phụ kiện cho trẻ em và trẻ sơ sinh",
    6: "Tã, Bỉm & Dụng cụ vệ sinh",
    7: "Xe, Ghế & Đai địu",
    8: "Đồ dùng phòng ngủ cho bé",
    9: "Đồ dùng phòng tắm cho bé",
    10: "Đồ dùng ăn dặm cho bé",
    11: "Đồ chơi cho bé",
    12: "Bộ boardgame",
    13: "Đồ chơi thẻ bài",
    14: "Búp bê",
    15: "Thú bông",
    16: "Phụ kiện búp bê, thú bông",
    17: "Hộp mù - Túi mù",
    18: "Mô hình giấy",
    19: "Mô hình 3D",
    20: "Xe mô hình",
    21: "Mô hình nhân vật",
    22: "Xe điều khiển & phụ kiện",
    23: "Máy bay điều khiển & phụ kiện",
    24: "Tàu điều khiển & phụ kiện",
    25: "Đồ chơi điều khiển khác",
    26: "Xe đạp cho bé",
    27: "Điện thoại",
    28: "Máy tính bảng",
    29: "Máy đọc sách",
    30: "Tivi",
    31: "Phụ kiện Tivi",
    32: "Thiết bị âm thanh",
    33: "Laptop",
    34: "Máy tính để bàn",
    35: "Thiết bị chơi game",
    36: "Thiết bị văn phòng",
    37: "Máy ảnh, Máy quay",
    38: "Camera giám sát",
    39: "Máy bay camera và phụ kiện",
    40: "Màn hình",
    41: "Thiết bị đeo thông minh",
    42: "Thiết bị lưu trữ",
    43: "Phụ kiện máy ảnh, máy quay",
    44: "Phụ kiện máy tính, laptop",
    45: "Phụ kiện điện thoại và máy tính bảng",
    46: "Phụ kiện điện tử khác",
    47: "Linh kiện máy tính",
    48: "Linh kiện điện thoại, máy tính bảng",
    49: "Linh kiện điện tử khác"
}

if __name__ == "__main__":
    folder_path = "images"  # change this to your actual folder
    image_paths = get_image_paths_from_folder(folder_path)
    print("🖼️ Danh sách ảnh tìm được:", image_paths)

    base64_images = encode_images_to_base64(image_paths)
    print("\n=== Bắt đầu phân tích từ AI ===\n")
    result = call_with_retry(base64_images, category_examples)
    print("\n🟢 Kết quả phân tích:\n", result)
    print("\n=== Kết thúc ===")
