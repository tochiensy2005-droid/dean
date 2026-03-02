import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# TRANSFORM (PHẢI NORMALIZE)
# ========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder("train", transform=train_transform)
val_dataset = datasets.ImageFolder("val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)

# ========================
# MODEL
# ========================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# ===== GIAI ĐOẠN 1: Train classifier =====
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

epochs_stage1 = 10

print("\n=== STAGE 1: Training classifier ===\n")

for epoch in range(epochs_stage1):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"[Stage1] Epoch {epoch+1}/{epochs_stage1} - Loss: {running_loss:.4f} - Train Acc: {train_acc:.2f}%")

# ===== GIAI ĐOẠN 2: Fine-tune toàn bộ model =====
print("\n=== STAGE 2: Fine-tuning entire model ===\n")

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
epochs_stage2 = 10

for epoch in range(epochs_stage2):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"[Stage2] Epoch {epoch+1}/{epochs_stage2} - Loss: {running_loss:.4f} - Train Acc: {train_acc:.2f}%")

print("\nTraining complete!")

torch.save(model.state_dict(), "tourism_model.pth")
print("Model saved as tourism_model.pth")