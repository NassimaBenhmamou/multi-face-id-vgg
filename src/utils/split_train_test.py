# src/utils/split_train_test.py

import os
import shutil
import random

source_dir = 'data/processed/'
train_dir = 'data/trTain/'
test_dir = 'data/test/'
split_ratio = 0.8  # 80% train, 20% test

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)
    split_point = int(len(images) * split_ratio)

    train_images = images[:split_point]
    test_images = images[split_point:]

    # Créer dossiers train/test
    create_dir(os.path.join(train_dir, class_name))
    create_dir(os.path.join(test_dir, class_name))

    # Copier les images
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

print("✅ Séparation train/test terminée.")
