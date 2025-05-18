import torch
from dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from model import EfficientNetB0Classifier
from tqdm import tqdm
import pandas as pd

# 1. 加载模型
model = EfficientNetB0Classifier(num_classes=18)
model.load_state_dict(torch.load('efficientnet_b0_multilabel.pth', map_location='cpu'))
model.eval()

# 2. 加载test数据
test_dataset = MultiModalDataset('test.csv', img_dir='/home/shuozhang/COMP5329/data', mode='test', num_classes=18)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

all_preds = []
with torch.no_grad():
    for images, _, _ in tqdm(test_loader):
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        all_preds.append(preds.cpu())
all_preds = torch.cat(all_preds, dim=0).numpy()

# 3. 还原标签编号
label_map = [i for i in range(1, 20) if i != 12]
test_df = test_dataset.df
results = []
for i, row in enumerate(all_preds):
    labels = [str(label_map[j]) for j, v in enumerate(row) if v == 1]
    results.append(' '.join(labels))
test_df['Labels'] = results
submission = test_df[['ImageID', 'Labels']]
submission.to_csv('submission_image.csv', index=False)
print('Kaggle提交文件 submission_image.csv 已生成！')