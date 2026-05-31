# Evaluate Trained ResNet50 Model

import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------
# Configuration
# ----------------------------------------

DATA_PATH = "flood_detection_split"
MODEL_PATH = "resnet50_flood_damage.pth"

# ----------------------------------------
# Device Configuration
# ----------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ----------------------------------------
# Image Preprocessing
# ----------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ----------------------------------------
# Load Test Dataset
# ----------------------------------------

test_dataset = datasets.ImageFolder(
    f"{DATA_PATH}/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

# ----------------------------------------
# Load Model
# ----------------------------------------

model = models.resnet50()

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

# ----------------------------------------
# Evaluation
# ----------------------------------------

y_true = []
y_pred = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        _, predictions = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(
            predictions.cpu().numpy()
        )

# ----------------------------------------
# Classification Report
# ----------------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Damaged",
            "Non-Damaged"
        ]
    )
)

# ----------------------------------------
# Confusion Matrix
# ----------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Damaged",
        "Non-Damaged"
    ],
    yticklabels=[
        "Damaged",
        "Non-Damaged"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved.")
