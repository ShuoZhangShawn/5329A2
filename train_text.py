import torch
from dataloader import MultiModalDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from model_text import TextClassifier
from tqdm import tqdm

# 1. 一次性读取全部数据
full_dataset = MultiModalDataset('train.csv', img_dir=None, mode='train', num_classes=18)

# 2. 划分训练集和验证集索引
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

# 3. 构建Subset和DataLoader
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TextClassifier(num_classes=18).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    
torch.save(model.state_dict(), 'tinybert_text_classifier.pth')