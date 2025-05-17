import torch
import torch.nn as nn
from transformers import AutoModel

class TextClassifier(nn.Module):
    def __init__(self, num_classes=18, pretrained_model='huawei-noah/TinyBERT_General_4L_312D'):  # TinyBERT官方权重
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取[CLS] token的输出
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        logits = self.classifier(cls_output)
        return logits

if __name__ == '__main__':
    model = TextClassifier(num_classes=18)
    print(model)