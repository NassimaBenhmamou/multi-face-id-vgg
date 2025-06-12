import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# --- Chargement des embeddings d'entraînement ---
embeddings_path = 'models/arcface/embeddings.pkl'
with open(embeddings_path, 'rb') as f:
    embedding_dict = pickle.load(f)

# Fusionner tous les embeddings train dans un tableau et créer un label associé
train_embeddings = []
train_labels = []
for person, embeds in embedding_dict.items():
    train_embeddings.append(embeds)  # embeds.shape = (N_img, 512)
    train_labels.extend([person]*embeds.shape[0])
train_embeddings = np.vstack(train_embeddings)  # (total_train_images, 512)
train_labels = np.array(train_labels)

# --- Initialisation du modèle ArcFace ---
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))

# --- Fonction pour extraire l'embedding ArcFace d'une image ---
def get_arcface_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    faces = app.get(img_np)
    if len(faces) == 0:
        print(f"Aucun visage détecté dans {image_path}")
        return None
    # On prend le premier visage détecté
    face = faces[0]
    return face.embedding

# --- Evaluation sur le dossier test ---
test_dir = 'data/test/'
y_true = []
y_pred = []

for person in tqdm(os.listdir(test_dir)):
    person_dir = os.path.join(test_dir, person)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        emb = get_arcface_embedding(img_path)
        if emb is None:
            continue

        # Calculer la similarité cosinus avec tous les embeddings train
        sims = cosine_similarity(emb.reshape(1, -1), train_embeddings)  # (1, total_train_images)
        best_idx = np.argmax(sims)
        pred_person = train_labels[best_idx]

        y_true.append(person)
        y_pred.append(pred_person)

# --- Calcul de l'accuracy ---
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"Accuracy ArcFace sur test : {accuracy*100:.2f}%")
