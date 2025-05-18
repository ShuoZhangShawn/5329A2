import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 加载ImageNet预训练权重
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # 替换分类头
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    @staticmethod
    def extractor(out_dim=1280):
        # EfficientNet-B0最后的特征维度是1280
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # 去掉分类头，只保留特征部分
        features = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
        )
        return features

