import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Configuration
data_dir = 'data/test/'
model_path = 'models/custom/custom_model.pth'
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Prétraitement
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# === Dataset & DataLoader
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
class_names = dataset.classes

# === Chargement du modèle
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Évaluation
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\n✅ Accuracy on test set: {accuracy:.2f}%")

# === Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
print("\n✅ Confusion Matrix:")
print(cm)

# === Affichage graphique
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.title(f'Matrice de Confusion - Accuracy: {accuracy:.2f}%')
plt.tight_layout()
plt.show()
