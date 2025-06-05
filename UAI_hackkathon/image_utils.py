import base64
import os
import csv

def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_images_to_base64(file_paths: list[str]) -> list[str]:
    return [encode_image_to_base64(path) for path in file_paths]

def get_image_paths_from_folder(folder_path: str, extensions={".jpg", ".jpeg", ".png", ".webp"}) -> list:
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.splitext(file)[1].lower() in extensions
    ]
def read_csv_with_csv_module(file_path: str) -> list[dict]:
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]

def load_categories(file_path: str) -> list[dict]:
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader if any(row.values())]

def match_category(name_or_description: str, categories: list[dict]) -> list[str]:
    name_or_description = name_or_description.lower()
    for row in categories:
        # Ghép các cấp category thành 1 chuỗi và so khớp đơn giản
        full_category = " > ".join(filter(None, row.values())).lower()
        if any(word in name_or_description for word in full_category.split(" > ")):
            return [row.get(f"level_{i+1}", "") for i in range(5) if row.get(f"level_{i+1}", "")]
    return []

def load_category_examples_from_csv(file_path: str, max_lines: int = 30) -> str:
    category_lines = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Ghép các cấp có giá trị, bỏ cấp trống
            levels = [row.get(f"level_{i+1}", "").strip() for i in range(5)]
            levels = [lvl for lvl in levels if lvl]
            if levels:
                category_lines.append(f"- {' > '.join(levels)}")
            if len(category_lines) >= max_lines:
                break
    return "\n".join(category_lines)