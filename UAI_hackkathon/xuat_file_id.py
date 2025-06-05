import pandas as pd

# Đọc file CSV
df = pd.read_csv("categories.csv")

# Tạo dict lưu ID cho từng danh mục
level_ids = {}
id_counter = 1

def get_or_create_id(name):
    global id_counter
    if pd.isna(name):
        return None
    if name not in level_ids:
        level_ids[name] = id_counter
        id_counter += 1
    return level_ids[name]

# Thêm các cột ID tương ứng
for level in ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']:
    df[f'{level}_id'] = df[level].apply(get_or_create_id)

# Xuất file mới
df.to_csv("category_tree_with_level_ids.csv", index=False)