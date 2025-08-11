import pickle
from collections import defaultdict

# 加载数据库 info
with open('data/kl/v1.0-trainval/kl_dbinfos.pkl', 'rb') as f:
    db_infos = pickle.load(f)

# 统计每个类别的样本数
cls_counts = {cls: len(db_infos[cls]) for cls in db_infos}

# 排序输出
for cls, count in sorted(cls_counts.items(), key=lambda x: -x[1]):
    print(f"{cls}: {count}")
