import os
import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# === Chargement du modèle FaceNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').to(device).eval()

# === Chargement des embeddings d'entraînement
with open('models/facenet/embeddings.pkl', 'rb') as f:
    train_embeddings = pickle.load(f)

# === Transformation des images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Dossier test
test_dir = 'data/test/'

correct = 0
total = 0

for person in os.listdir(test_dir):
    person_dir = os.path.join(test_dir, person)
    if not os.path.isdir(person_dir):
        continue

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                test_embedding = model(img_tensor).cpu().numpy()

            # === Comparer avec les embeddings du train
            best_match = None
            highest_similarity = -1

            for train_person, embeddings in train_embeddings.items():
                similarities = cosine_similarity(test_embedding, embeddings)
                mean_similarity = np.mean(similarities)

                if mean_similarity > highest_similarity:
                    highest_similarity = mean_similarity
                    best_match = train_person

            # === Vérifier la prédiction
            total += 1
            if best_match == person:
                correct += 1

            print(f"Image: {image_name} | Vrai: {person} | Prédit: {best_match} | Similarité: {highest_similarity:.4f}")

        except Exception as e:
            print(f"[FaceNet] ⚠️ Erreur avec {image_path} : {e}")

# === Résultat final
accuracy = 100 * correct / total if total > 0 else 0
print(f"\n🎯 Accuracy sur le jeu de test : {accuracy:.2f}% ({correct}/{total})")
