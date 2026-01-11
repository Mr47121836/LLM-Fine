from modelscope.msdatasets import MsDataset
import json
import random

# 设置随机种子，保证每次运行结果可复现
random.seed(42)

# 从指定路径加载数据集（jsonl 格式）
# subset_name='default' 表示使用默认子集
# split='train' 这里只是命名习惯，实际上加载的是完整数据
ds = MsDataset.load(
    '../datasets/enhanced_network_dataset.jsonl',
    subset_name='default',
    split='train'
)

# 把数据集转换成 Python 列表，方便后续操作
data_list = list(ds)

# 随机打乱数据顺序（非常重要，避免数据原始顺序带来的偏差）
random.shuffle(data_list)

# 计算训练集分割点：前90%作为训练集
split_idx = int(len(data_list) * 0.9)

# 进行数据分割
train_data = data_list[:split_idx]      # 前 90% → 训练集
val_data = data_list[split_idx:]        # 后 10% → 验证集

# 将训练集保存到 train.jsonl 文件
with open('../datasets/train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        # 写入每条数据为 JSON 格式，确保中文不被转成 unicode 转义
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')  # 每条数据后换行（jsonl 格式要求）

# 将验证集保存到 val.jsonl 文件
with open('../datasets/val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# 输出分割结果信息
print("数据集分割成功完成！")
print(f"训练集大小：{len(train_data):,} 条")
print(f"验证集大小：{len(val_data):,} 条")