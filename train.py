from model import EfficientNetB0Classifier
import torch
from torch.utils.data import DataLoader
from dataloader import MultiModalDataset
from tqdm import tqdm

# 1. 配置参数
num_classes = 18
#mini batch
batch_size = 128
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 实例化模型、损失函数、优化器
model = EfficientNetB0Classifier(num_classes).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. 数据加载器
train_dataset = MultiModalDataset(
    csv_file='train.csv',
    img_dir='/home/shuozhang/COMP5329/data',
    mode='train'
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 4. 训练主循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # tqdm包裹train_loader，显示进度条
    for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 每个batch输出一次loss
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 5. 保存模型
torch.save(model.state_dict(), 'efficientnet_b0_multilabel.pth')
