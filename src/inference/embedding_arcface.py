import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
import insightface
from insightface.app import FaceAnalysis

# === Initialisation d'ArcFace (iresnet100)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # ou 'CUDAExecutionProvider'
app.prepare(ctx_id=0, det_size=(160, 160))  # Resize détecteur visage

# === Dossier contenant les visages
data_dir = 'data/train/'
embedding_dict = {}

# === Transformation pour PIL -> np.array
to_numpy = transforms.Compose([
    transforms.Resize((112, 112)),  # requis par ArcFace
])

# === Génération des embeddings
for person in tqdm(os.listdir(data_dir), desc="📦 Génération des embeddings ArcFace"):
    person_dir = os.path.join(data_dir, person)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for image_name in os.listdir(person_dir):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(person_dir, image_name)
        try:
            img = Image.open(image_path).convert('RGB')
            img = to_numpy(img)
            img = np.asarray(img)

            faces = app.get(img)
            if faces:
                # Prend le premier visage détecté
                embedding = faces[0].embedding
                embeddings.append(embedding)
            else:
                print(f"[ArcFace] ⚠️ Aucun visage détecté : {image_path}")

        except Exception as e:
            print(f"[ArcFace] ⚠️ Erreur avec {image_path} : {e}")

    if embeddings:
        embedding_dict[person] = np.vstack(embeddings)

# === Sauvegarde
output_path = 'models/arcface/embeddings.pkl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(embedding_dict, f)

print(f"[ArcFace] ✅ Embeddings enregistrés dans : {output_path}")
