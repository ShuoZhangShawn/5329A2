import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import re
from io import StringIO

class MultiModalDataset(Dataset):
    #定义元数据
    def __init__(self, csv_file, img_dir, num_classes=19, tokenizer_name='distilbert-base-uncased', mode='train', max_length=32):
        # 健壮读取csv，处理caption中的英文引号
        with open(csv_file) as file:
            lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
            self.df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
        self.img_dir = img_dir
        self.mode = mode
        self.num_classes = num_classes
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        # 标签编号1-19，12缺失，建立映射
        self.label_map = [i for i in range(1, 20) if i != 12]
        self.label2idx = {l: i for i, l in enumerate(self.label_map)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['ImageID']))
        image = Image.open(img_path).convert('RGB')
        # 简单resize+ToTensor，可根据需要自定义transform
        image = image.resize((224, 224))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        caption = str(row['Caption'])
        encoding = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        if self.mode == 'train':
            labels = str(row['Labels']).split()
            multi_hot = torch.zeros(self.num_classes)
            for l in labels:
                l = int(l)
                if l in self.label2idx:
                    multi_hot[self.label2idx[l]] = 1
            return image, input_ids, attention_mask, multi_hot
        else:
            return image, input_ids, attention_mask 


# 训练集
train_dataset = MultiModalDataset(
    csv_file='train.csv',
    img_dir='/home/shuozhang/COMP5329/data',
    mode='train'
)
img, input_ids, attention_mask, multi_hot = train_dataset[0]
print(img.shape, input_ids.shape, attention_mask.shape, multi_hot)

# 测试集
test_dataset = MultiModalDataset(
    csv_file='test.csv',
    img_dir='/home/shuozhang/COMP5329/data',
    mode='test'
)
img, input_ids, attention_mask = test_dataset[0]
print(img.shape, input_ids.shape, attention_mask.shape)

