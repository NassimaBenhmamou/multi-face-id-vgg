# src/visualization/plot_embeddings.py

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_tsne_embeddings(embeddings, labels, title="t-SNE des embeddings", save_path=None):
    print("[Visualisation] t-SNE en cours...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        idxs = labels == label
        plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], 
                    c=[colors(i)], label=label, alpha=0.7)
        
    plt.legend()
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"[Visualisation] Graphique sauvegardé dans : {save_path}")
    plt.show()
