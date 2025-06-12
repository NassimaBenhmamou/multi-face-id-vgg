import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# === Transformation d'image pour Facenet
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_facenet_embeddings():
    """Charge les embeddings et noms depuis un fichier pickle."""
    with open("models/facenet/embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    embeddings = list(data.values())  # liste d'embeddings (np.ndarray)
    names = list(data.keys())         # noms des personnes
    return embeddings, names

def extract_facenet_embedding(model, image: Image.Image):
    """Extrait l'embedding Facenet à partir d'une image PIL."""
    model.eval()
    img_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        embedding = model(img_tensor)  # shape: (1, emb_dim)

    return embedding.cpu().numpy()

def predict_facenet(model, image: Image.Image, embeddings_db, names):
    """Prédit la personne correspondant à l'image à partir des embeddings existants."""
    emb = extract_facenet_embedding(model, image)  # shape: (1, dim)
    best_match = None
    best_score = -1

    for i, stored_emb in enumerate(embeddings_db):
        # stored_emb shape: (N, dim) ou (dim,)
        if stored_emb.ndim == 1:
            stored_emb = stored_emb.reshape(1, -1)

        sims = cosine_similarity(emb, stored_emb)
        max_sim = np.max(sims)

        if max_sim > best_score:
            best_score = max_sim
            best_match = names[i]

    if best_score > 0.5:
        return best_match, best_score * 100
    else:
        return None, 0.0
