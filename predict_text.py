import torch
from dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from model_text import TextClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd

# 加载模型
model = TextClassifier(num_classes=18)
model.load_state_dict(torch.load('tinybert_text_classifier.pth', map_location='cpu'))
model.eval()

# 加载test数据
test_dataset = MultiModalDataset('test.csv', img_dir=None, mode='test', num_classes=18)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

all_preds = []
with torch.no_grad():
    for input_ids, attention_mask in tqdm(test_loader):
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs)  # [batch, 18]
        preds = (probs > 0.5).int()     # multi-hot
        all_preds.append(preds.cpu())
all_preds = torch.cat(all_preds, dim=0).numpy()
# all_preds: [num_samples, 18]，每一行为multi-hot预测

# ===== 生成Kaggle要求的submission.csv文件 =====
label_map = [i for i in range(1, 20) if i != 12]
#复用MultiModalDataset的健壮读取结果
#你可以直接用test_dataset.df，而不是再用pd.read_csv('test.csv')，这样就不会报错了。
test_df = test_dataset.df
results = []
for i, row in enumerate(all_preds):
    labels = [str(label_map[j]) for j, v in enumerate(row) if v == 1]
    results.append(' '.join(labels))
test_df['Labels'] = results
# 保证只保留ImageID和Labels两列，且无多余空格
submission = test_df[['ImageID', 'Labels']]
submission.to_csv('submission_text.csv', index=False)
print('Kaggle提交文件 submission.csv 已生成！')