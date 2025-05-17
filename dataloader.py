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
    def __init__(self, csv_file, img_dir, num_classes=18, tokenizer_name='distilbert-base-uncased', mode='train', max_length=32):
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
        # 只做文本
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
            if self.img_dir is not None:
                # 需要图片时
                img_path = os.path.join(self.img_dir, str(row['ImageID']))
                image = Image.open(img_path).convert('RGB')
                image = image.resize((224, 224))
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                return image, input_ids, attention_mask, multi_hot
            else:
                # 只文本
                return input_ids, attention_mask, multi_hot
        else:
            if self.img_dir is not None:
                img_path = os.path.join(self.img_dir, str(row['ImageID']))
                image = Image.open(img_path).convert('RGB')
                image = image.resize((224, 224))
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                return image, input_ids, attention_mask
            else:
                return input_ids, attention_mask


"""
image
类型：torch.Tensor
形状：[3, 224, 224]
内容：一张图片的像素数据，已经被 resize 到 224x224，3个通道（RGB），并归一化到0~1之间。
用途：作为视觉模型（如MobileNetV2、ResNet等）的输入。
"""

"""
input_ids
类型：torch.Tensor
形状：[max_length]
内容：caption的tokenized结果，max_length=32。
用途：作为语言模型（如BERT、RoBERTa等）的输入。
"""

"""
attention_mask
类型：torch.Tensor
形状：[max_length]
内容：与input_ids相同，表示哪些token是实际内容，哪些是padding。
用途：告诉模型哪些位置是实际内容，哪些是padding。
"""

"""
multi_hot
类型：torch.Tensor
形状：[num_classes]
内容：一个多热编码向量，表示图片的标签。
用途：作为分类模型的输出。
"""

