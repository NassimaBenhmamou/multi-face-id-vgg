import os
import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# === Initialisation du modèle FaceNet préentraîné
model = InceptionResnetV1(pretrained='vggface2').eval()

# === Transformation des images (MTCNN output ready)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1] pour FaceNet
])

# === Dossier à parcourir : uniquement les données d'entraînement
data_dir = 'data/train/'
embedding_dict = {}

# === Génération des embeddings
for person in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # (1, 3, 160, 160)
            with torch.no_grad():
                embedding = model(img_tensor).squeeze(0).numpy()
            embeddings.append(embedding)
        except Exception as e:
            print(f"[FaceNet] ⚠️ Erreur avec {image_path} : {e}")

    if embeddings:
        embedding_dict[person] = np.vstack(embeddings)

# === Sauvegarde du dictionnaire d'embeddings
output_path = 'models/facenet/embeddings.pkl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(embedding_dict, f)

print(f"[FaceNet] ✅ Embeddings enregistrés dans : {output_path}")
