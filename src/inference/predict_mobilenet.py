import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models
import torch.nn as nn

def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 6)  # ← Mettez le bon nombre de classes ici
    model.load_state_dict(torch.load("models/custom/custom_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


def extract_mobilenet_embedding(model, image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(img_tensor)  # shape (1, 128)

    return embedding.cpu().numpy()

def predict_mobilenet(model, image: Image.Image, embeddings_db, names):
    emb = extract_mobilenet_embedding(model, image)
    best_match = None
    best_score = -1

    for i, stored_emb in enumerate(embeddings_db):
        if len(stored_emb.shape) == 1:
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
