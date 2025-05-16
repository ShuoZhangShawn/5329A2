import torch
import torch.nn as nn
import open_clip
from PIL import Image
from torchvision import transforms

# 假设你的类别数
num_classes = 20

# 1. 加载CLIP模型（以ViT-B-32为例）
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 2. 冻结CLIP参数（可选，微调时可解冻部分层）
for param in clip_model.parameters():
    param.requires_grad = False

# 3. 定义多标签分类头
class MultiModalClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, text):
        # 提取图片和文本特征
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(text)
        # 拼接特征
        fused = torch.cat([image_features, text_features], dim=1)
        # 分类
        return self.classifier(fused)

# 4. 实例化模型
embed_dim = clip_model.visual.output_dim  # 通常为512
model = MultiModalClassifier(clip_model, embed_dim, num_classes)

# 5. 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

# 6. 示例数据
# 假设有一张图片和一个caption
img = Image.open('your_image.jpg').convert('RGB')
caption = "A cat sitting on a sofa."

# 预处理
image_tensor = preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]
text_tensor = tokenizer([caption])           # [1, seq_len]

# 7. 前向传播
outputs = model(image_tensor, text_tensor)
labels = torch.randint(0, 2, (1, num_classes)).float()  # 随机多标签

loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

print("训练一步完成，loss:", loss.item())