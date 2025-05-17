import re
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from collections import Counter

FILENAME = 'train.csv'
with open(FILENAME) as file:
    # 处理 caption 字段中的英文引号问题
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")

# 标签分布统计
all_labels = []
for labels in df['Labels']:
    all_labels.extend(str(labels).split())

label_count = Counter(all_labels)
print("标签分布：", label_count)

# 可视化
labels = list(label_count.keys())
counts = [label_count[l] for l in labels]

plt.figure(figsize=(10,5))
plt.bar(labels, counts)
plt.xlabel('label class')
plt.ylabel('Frequency')
plt.title('Label class distribution')
plt.show()



text = []
for i in range(100):
    text = df['Caption'][i]
    text.append(text)

print(text)

#从EDA可以看到 类别1 有很大占比，也就是人像占很大比例。 也就是说，我们需要找一个擅长人像的预训练模型