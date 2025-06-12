# Multi-Model Face Recognition App 🎯

Une application avancée de reconnaissance faciale en temps réel (Streamlit) combinant trois modèles puissants : **ArcFace**, **FaceNet** et un **MobileNet personnalisé**.

## 🚀 Fonctionnalités principales

- 📸 Détection de visages avec MTCNN (robuste aux angles, expressions et luminosité)
- 🧠 Prédiction d'identité avec :
  - **ArcFace** (pré-entraîné)
  - **FaceNet** (pré-entraîné)
  - **MobileNet** (personnalisé sur VGGFace2)
- 🔄 Mode webcam live ou téléversement d’image
- 🧾 Affichage simultané des prédictions de chaque modèle avec score de confiance
- 🗂️ Historique local des prédictions
- 🎯 Compatible GPU (CUDA) et CPU

## 🛠️ Stack technique

- Python 3.10+
- Streamlit
- PyTorch
- torchvision
- scikit-learn
- facenet-pytorch



