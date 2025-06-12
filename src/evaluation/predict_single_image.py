# src/prediction/predict_single_image.py

import torch
from torchvision import transforms, models
from PIL import Image
import os

# Paramètres
image_path = 'data/0101_01.jpg'  # ➜ change avec ton image
model_path = 'models/custom/custom_model.pth'
class_names = sorted(os.listdir('data/processed/'))  # noms des classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prétraitement
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Chargement image
img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# Modèle
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Prédiction
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]

print(f"Predicted person: {predicted_class}")
