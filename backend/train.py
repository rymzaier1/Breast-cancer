import os
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.metrics import accuracy_score

from model import BreastCancerModel

print("Starting training...")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ==========================================
# TRANSFORMS
# ==========================================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),

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

# ==========================================
# DATASETS
# ==========================================

train_dataset = datasets.ImageFolder(
    "../dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "../dataset/val",
    transform=val_transform
)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

# ==========================================
# MODEL
# ==========================================

model = BreastCancerModel().to(device)

# IMPORTANT
class_weights = torch.tensor([1.0, 2.0]).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = Adam(
    model.parameters(),
    lr=0.0001
)

best_acc = 0

# ==========================================
# TRAINING
# ==========================================

for epoch in range(15):

    print(f"\nEpoch {epoch+1}")

    model.train()

    running_loss = 0

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 20 == 0:
            print(f"Batch {i}")

    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    acc = accuracy_score(targets, preds)

    print(f"Validation Accuracy: {acc:.4f}")

    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if acc > best_acc:

        best_acc = acc

        os.makedirs("../models", exist_ok=True)

        torch.save(
            model.state_dict(),
            "../models/best_model.pth"
        )

        print("Best model saved!")

print("\nTraining finished!")