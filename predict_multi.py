import torch
from dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from model_muti import MultiModalClassifier
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalClassifier(num_classes=18).to(device)
model.load_state_dict(torch.load('multimodal_classifier.pth', map_location=device))
model.eval()

test_dataset = MultiModalDataset('test.csv', img_dir='/home/shuozhang/COMP5329/data', mode='test', num_classes=18)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

all_preds = []
with torch.no_grad():
    for images, input_ids, attention_mask in tqdm(test_loader):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(images, input_ids, attention_mask)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        all_preds.append(preds.cpu())
all_preds = torch.cat(all_preds, dim=0).numpy()

label_map = [i for i in range(1, 20) if i != 12]
test_df = test_dataset.df
results = []
for i, row in enumerate(all_preds):
    labels = [str(label_map[j]) for j, v in enumerate(row) if v == 1]
    results.append(' '.join(labels))
test_df['Labels'] = results
submission = test_df[['ImageID', 'Labels']]
submission.to_csv('submission_multi.csv', index=False)
print('Kaggle提交文件 submission_multi.csv 已生成！')
