import sys
import os

# Ajoute le chemin racine au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# inference.py
import torch
from torchvision import transforms
from PIL import Image
import os

from src.models.custom_model import MyModel  # assure-toi que le chemin est correct

# === Config
model_path = 'models/custom/custom_model.pth'
data_dir = 'data/train'  # pour récupérer les noms de classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(os.listdir(data_dir))

# === Chargement du modèle
model = MyModel(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Transformation de l’image
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# === Exemple d’image à prédire
img_path = 'data/test/person1.jpg'  # image à tester
image = Image.open(img_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)

# === Prédiction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

class_names = os.listdir(data_dir)  # ['person1', 'person2', ...]
print(f"Classe prédite : {class_names[predicted.item()]}")
