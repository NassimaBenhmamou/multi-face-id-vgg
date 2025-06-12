import os
import pandas as pd
from mtcnn import MTCNN
import numpy as np
from PIL import Image
from tqdm import tqdm

# Initialisation MTCNN
detector = MTCNN()

# Chemins
csv_path = 'data/raw/identity_meta.csv'
img_dir = 'data/raw/train'
output_dir = 'data/processed/'

# Lire le CSV avec nettoyage
df = pd.read_csv(csv_path, skipinitialspace=True)

# Affichage debug des colonnes
print("Colonnes disponibles :", df.columns.tolist())

# Pour chaque personne dans le CSV avec barre de progression
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Prétraitement des images"):
    try:
        vgg_id = row['Class_ID']
        name = str(row['Name']).replace('"', '').replace(" ", "_")

        # Chemins
        person_dir = os.path.join(img_dir, vgg_id)
        output_person_dir = os.path.join(output_dir, name)
        os.makedirs(output_person_dir, exist_ok=True)

        # Vérification de l'existence du dossier source
        if not os.path.exists(person_dir):
            print(f"[Attention] Dossier introuvable : {person_dir}")
            continue

        # Parcourir toutes les images du dossier
        for img_name in os.listdir(person_dir):
            # Filtrer extensions valides
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(person_dir, img_name)

            try:
                img = Image.open(img_path).convert('RGB')
                faces = detector.detect_faces(np.array(img))

                if faces:
                    x, y, w, h = faces[0]['box']
                    x, y = max(0, x), max(0, y)
                    face = img.crop((x, y, x + w, y + h)).resize((160, 160))
                    face.save(os.path.join(output_person_dir, img_name))

            except Exception as e:
                print(f"[Erreur image] {img_name} : {e}")

    except Exception as e:
        print(f"[Erreur ligne CSV {idx}] : {e}")
