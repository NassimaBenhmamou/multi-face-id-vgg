# src/inference/predict_arcface.py
import os
import pickle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# === Initialiser ArcFace
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # ou 'CUDAExecutionProvider'
face_app.prepare(ctx_id=0, det_size=(160, 160))

# === Prétraitement image
to_numpy = transforms.Compose([
    transforms.Resize((112, 112)),
])

def load_model_and_db():
    """Charge les embeddings sauvegardés et initialise ArcFace."""
    db_path = "models/arcface/embeddings.pkl"
    with open(db_path, "rb") as f:
        embedding_dict = pickle.load(f)
    
    names = list(embedding_dict.keys())
    return face_app, embedding_dict, names

def predict(model, image: Image.Image, db: dict, names: list):
    """Prédit l'identité à partir d'une image avec ArcFace."""
    image = to_numpy(image)
    img_np = np.asarray(image)

    faces = model.get(img_np)
    if not faces:
        return None

    emb = faces[0].embedding.reshape(1, -1)

    best_match = None
    best_score = -1

    for name in names:
        stored_embeddings = db[name]
        sims = cosine_similarity(emb, stored_embeddings)
        max_sim = np.max(sims)
        if max_sim > best_score:
            best_score = max_sim
            best_match = name

    if best_score > 0.5:
        return best_match, best_score * 100
    else:
        return None
