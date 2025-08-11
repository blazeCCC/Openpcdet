import os
import json

# label_root = "/home/baojiali/Downloads/disk1/data/lightwheel_data_clean/label"
label_root = "/home/baojiali/Downloads/disk1/data/lightwheel_data_clean_wheelcrane"
target_subtypes = ["Crane", "ContainerForklift","WheelCrane"]

# 初始化计数器
label_file_count = 0
subtype_counts = {subtype: 0 for subtype in target_subtypes}

# 遍历所有 label json 文件
for root, _, files in os.walk(label_root):
    for file in files:
        if not file.endswith(".json"):
            continue

        label_file_count += 1
        json_path = os.path.join(root, file)

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            for obj in data:
                subtype = obj.get("subtype")
                if subtype in subtype_counts:
                    subtype_counts[subtype] += 1
        except Exception as e:
            print(f"Failed to read {json_path}: {e}")

# ✅ 输出结果
print(f"Total label files: {label_file_count}")
print("Subtype counts:")
for subtype, count in subtype_counts.items():
    print(f"  {subtype}: {count}")
