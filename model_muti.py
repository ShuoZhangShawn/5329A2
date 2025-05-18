import torch
import torch.nn as nn
from model import EfficientNetB0Classifier
from model_text import TextClassifier

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=18, img_feat_dim=1280, text_feat_dim=312, hidden_dim=512):
        super().__init__()
        # 视觉主干（去掉最后一层分类头，只要特征）
        self.vision_backbone = EfficientNetB0Classifier.extractor(img_feat_dim)  # 你需要在EfficientNetB0Classifier里加个extractor方法，返回特征
        # 文本主干（去掉最后一层分类头，只要特征）
        self.text_backbone = TextClassifier.extractor(text_feat_dim)  # 同理
        # 融合后的全连接分类头
        self.classifier = nn.Sequential(
            nn.Linear(img_feat_dim + text_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.vision_backbone(image)           # [batch, img_feat_dim]
        text_feat = self.text_backbone(input_ids, attention_mask)  # [batch, text_feat_dim]
        fused = torch.cat([img_feat, text_feat], dim=1)  # [batch, img_feat_dim+text_feat_dim]
        if not hasattr(self, 'print_count'):
            self.print_count = 0
        if self.print_count < 3:
            print('拼接后向量 shape:', fused.shape)
            print('拼接后向量前2个样本:', fused[:2])
            print('img_feat shape:', img_feat.shape)
            print('text_feat shape:', text_feat.shape)
            print('fused shape:', fused.shape)
            self.print_count += 1
        out = self.classifier(fused)
        return out
