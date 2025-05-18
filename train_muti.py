import torch
from dataloader import MultiModalDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from model_muti import MultiModalClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. 一次性读取全部数据
full_dataset = MultiModalDataset('train.csv', img_dir='/home/shuozhang/COMP5329/data', mode='train', num_classes=18)

# 2. 划分训练集和验证集索引
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

# 3. 构建Subset和DataLoader
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

batch_size = 64
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model = MultiModalClassifier(num_classes=18).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#可视化loss
train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    # 训练
    model.train()
    train_loss = 0
    for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    #可视化loss
    train_loss_list.append(train_loss/len(train_loader))

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    #可视化loss
    val_loss_list.append(val_loss/len(val_loader))

torch.save(model.state_dict(), 'multimodal_classifier.pth')
#可视化训练过程
plt.figure()
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.savefig('loss_curve.png')  # 保存为图片
plt.show()  # 或者直接显示
