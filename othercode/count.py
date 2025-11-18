import json

# 1. 指定 JSON 文件路径
json_file_path = "model/sgcombined.json"  # 改成你的 JSON 文件名

# 2. 读取 JSON 文件
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. 统计顶层有多少个键
num_keys = len(data)
print(f"JSON 文件中共有 {num_keys} 个键。")

# 4. 可选：打印所有键名
#print("键名列表：", list(data.keys()))
