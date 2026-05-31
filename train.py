# train.py

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ----------------------------------------
# Configuration
# ----------------------------------------

DATA_PATH = "flood_detection_split"
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20

# ----------------------------------------
# Device Configuration
# ----------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# ----------------------------------------
# Image Preprocessing
# ----------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],   # ImageNet mean
        [0.229, 0.224, 0.225]    # ImageNet std
    )
])

# ----------------------------------------
# Load Dataset
# ----------------------------------------

train_dataset = datasets.ImageFolder(
    f"{DATA_PATH}/train",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Classes:", train_dataset.class_to_idx)

# ----------------------------------------
# Load Pretrained ResNet50
# ----------------------------------------

model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# Freeze feature extraction layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(device)

# ----------------------------------------
# Training Setup
# ----------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# ----------------------------------------
# Training Loop
# ----------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        _, predictions = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)

        correct += torch.sum(
            predictions == labels.data
        )

    epoch_loss = running_loss / len(train_dataset)

    epoch_accuracy = (
        correct.double() / len(train_dataset)
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_accuracy:.4f}"
    )

# ----------------------------------------
# Save Model
# ----------------------------------------

torch.save(
    model.state_dict(),
    "resnet50_flood_damage.pth"
)

print("Model saved successfully.")
