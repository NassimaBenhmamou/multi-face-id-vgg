import streamlit as st
import os
import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2
import time
from torchvision import models
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(195deg, #42424a 0%, #191919 100%);
        color: white;
    }
    .widget-label {
        font-weight: 600 !important;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #0c0c0c;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        background-image: none;
        color: #000000;
    }
    .big-font {
        font-size:24px !important;
        font-weight: 700 !important;
    }
    .model-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .header-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .comparison-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #f54ea2 0%, #ff7676 100%);
    }
    .webcam-box {
        border: 3px solid #764ba2;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .prediction-box {
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        background: rgba(255, 255, 255, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# Load models (with caching)
@st.cache_resource
def load_facenet():
    model = InceptionResnetV1(pretrained='vggface2').eval()
    try:
        with open('models/facenet/embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        return model, embeddings
    except Exception as e:
        st.error(f"Error loading FaceNet: {e}")
        return None, None

@st.cache_resource
def load_arcface():
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(160, 160))
        with open('models/arcface/embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        return app, embeddings
    except Exception as e:
        st.error(f"Error loading ArcFace: {e}")
        return None, None

@st.cache_resource
def load_custom_model():
    try:
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, len(os.listdir('data/train/')))
        model.load_state_dict(torch.load('models/custom/custom_model.pth', map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading custom model: {e}")
        return None

# Initialize models
facenet_model, facenet_embeddings = load_facenet()
arcface_app, arcface_embeddings = load_arcface()
custom_model = load_custom_model()

# Transformations
facenet_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

custom_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Utility functions
def get_facenet_prediction(img):
    img_tensor = facenet_transform(img).unsqueeze(0)
    with torch.no_grad():
        query_embedding = facenet_model(img_tensor).squeeze(0).numpy()
    
    similarities = {}
    for name, embeddings in facenet_embeddings.items():
        sim = cosine_similarity([query_embedding], embeddings)
        similarities[name] = float(np.max(sim))
    
    predicted_name = max(similarities.items(), key=lambda x: x[1])[0]
    confidence = similarities[predicted_name]
    
    return predicted_name, confidence, similarities

def get_arcface_prediction(img):
    img = img.resize((112, 112))
    img_array = np.array(img)
    faces = arcface_app.get(img_array)
    
    if not faces:
        return None, 0, {}
    
    query_embedding = faces[0].embedding
    similarities = {}
    for name, embeddings in arcface_embeddings.items():
        sim = cosine_similarity([query_embedding], embeddings)
        similarities[name] = float(np.max(sim))
    
    predicted_name = max(similarities.items(), key=lambda x: x[1])[0]
    confidence = similarities[predicted_name]
    
    return predicted_name, confidence, similarities

def get_custom_model_prediction(img):
    img_tensor = custom_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = custom_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    class_names = sorted(os.listdir('data/train/'))
    conf, pred = torch.max(probabilities, 1)
    predicted_name = class_names[pred.item()]
    confidence = conf.item()
    
    # Get all confidences
    similarities = {class_names[i]: float(probabilities[0][i]) for i in range(len(class_names))}
    
    return predicted_name, confidence, similarities

def plot_confidences(confidences, title):
    df = pd.DataFrame.from_dict(confidences, orient='index', columns=['Confidence'])
    df = df.sort_values(by='Confidence', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=df.index, y='Confidence', data=df, palette='viridis', ax=ax)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    return fig

def display_confidence_bars(confidences):
    if not confidences:
        return
    
    # Trouver la meilleure prédiction
    best_name, best_conf = max(confidences.items(), key=lambda x: x[1])
    
    st.write(f"{best_name}:")
    st.markdown(f"""
    <div class="confidence-bar" style="width: {best_conf*100}%"></div>
    <small>{best_conf:.2%}</small>
    """, unsafe_allow_html=True)

# Main app
def main():
    st.sidebar.title("Configuration")
    option = st.sidebar.radio("Select Input Mode:", ("Upload Image", "Webcam"))
    
    st.sidebar.markdown("---")
    st.sidebar.header("Model Selection")
    use_facenet = st.sidebar.checkbox("Use FaceNet", True)
    use_arcface = st.sidebar.checkbox("Use ArcFace", True)
    use_custom = st.sidebar.checkbox("Use Custom Model", True)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.01)
    realtime_update = st.sidebar.slider("Realtime Update Interval (ms)", 100, 2000, 500, 100)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Face Recognition System**\n
    This application compares three face recognition models:
    - **FaceNet**: Based on InceptionResnetV1
    - **ArcFace**: State-of-the-art face recognition
    - **Custom Model**: Fine-tuned MobileNetV2
    """)
    
    # Main content
    st.markdown('<div class="header-card"><h1 style="color:white;">Face Recognition System</h1></div>', unsafe_allow_html=True)
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            # Display smaller image
            st.image(image, caption='Uploaded Image', width=300)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="model-card"><h3>Model Predictions</h3></div>', unsafe_allow_html=True)
                
                results = {}
                if use_facenet and facenet_model is not None:
                    with st.spinner('Running FaceNet...'):
                        start_time = time.time()
                        name, conf, all_conf = get_facenet_prediction(image)
                        results['FaceNet'] = {
                            'name': name,
                            'confidence': conf,
                            'time': time.time() - start_time,
                            'all_confidences': all_conf
                        }
                
                if use_arcface and arcface_app is not None:
                    with st.spinner('Running ArcFace...'):
                        start_time = time.time()
                        name, conf, all_conf = get_arcface_prediction(image)
                        results['ArcFace'] = {
                            'name': name,
                            'confidence': conf,
                            'time': time.time() - start_time,
                            'all_confidences': all_conf
                        }
                
                if use_custom and custom_model is not None:
                    with st.spinner('Running Custom Model...'):
                        start_time = time.time()
                        name, conf, all_conf = get_custom_model_prediction(image)
                        results['Custom Model'] = {
                            'name': name,
                            'confidence': conf,
                            'time': time.time() - start_time,
                            'all_confidences': all_conf
                        }
                
                # Display results
                for model_name, data in results.items():
                    st.markdown(f'<div class="model-card"><h4>{model_name}</h4>', unsafe_allow_html=True)
                    
                    if data['confidence'] >= confidence_threshold:
                        st.success(f"Predicted: {data['name']} (Confidence: {data['confidence']:.2%})")
                    else:
                        st.warning(f"Low Confidence Prediction: {data['name']} (Confidence: {data['confidence']:.2%})")
                    
                    st.write(f"Processing Time: {data['time']:.2f} seconds")
                    display_confidence_bars(data['all_confidences'])
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="comparison-card"><h3>Model Comparison</h3></div>', unsafe_allow_html=True)
                
                if len(results) > 0:
                    # Create comparison data
                    comparison_data = []
                    for model_name, data in results.items():
                        comparison_data.append({
                            'Model': model_name,
                            'Confidence': data['confidence'],
                            'Time (s)': data['time'],
                            'Prediction': data['name']
                        })
                    
                    df = pd.DataFrame(comparison_data)
                    st.table(df.style.background_gradient(cmap='Blues'))
                    
                    # Agreement analysis
                    predictions = [data['name'] for data in results.values()]
                    agreement = len(set(predictions)) == 1
                    
                    if agreement:
                        st.success("All models agree on the prediction!")
                    else:
                        st.warning("Models disagree on the prediction.")
                        st.write("Disagreement analysis:")
                        for model_name, data in results.items():
                            st.write(f"- {model_name}: {data['name']} ({data['confidence']:.2%})")
                    
                    # Confidence comparison chart
                    fig, ax = plt.subplots()
                    sns.barplot(x='Model', y='Confidence', data=df, palette='mako')
                    plt.title('Model Confidence Comparison')
                    plt.ylim(0, 1)
                    st.pyplot(fig)
                    
                    # Time comparison chart
                    fig, ax = plt.subplots()
                    sns.barplot(x='Model', y='Time (s)', data=df, palette='rocket')
                    plt.title('Processing Time Comparison')
                    st.pyplot(fig)
                else:
                    st.warning("No models selected or available for comparison.")
    
    else:  # Webcam mode
        st.markdown('<div class="model-card"><h3>Real-time Face Recognition</h3></div>', unsafe_allow_html=True)
        
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        last_update = 0
        current_results = {}
        
        # Create placeholders for model predictions
        pred_placeholders = {}
        if use_facenet:
            pred_placeholders['FaceNet'] = st.empty()
        if use_arcface:
            pred_placeholders['ArcFace'] = st.empty()
        if use_custom:
            pred_placeholders['Custom Model'] = st.empty()
        
        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time() * 1000  # milliseconds
            
            # Display frame
            FRAME_WINDOW.image(frame, channels="RGB", use_column_width=True)
            
            # Process frame at specified interval
            if now - last_update > realtime_update:
                last_update = now
                pil_image = Image.fromarray(frame)
                
                current_results = {}  # Reset results
                
                # Get predictions
                if use_facenet and facenet_model is not None:
                    name, conf, _ = get_facenet_prediction(pil_image)
                    current_results['FaceNet'] = {
                        'name': name,
                        'confidence': conf
                    }
                
                if use_arcface and arcface_app is not None:
                    name, conf, _ = get_arcface_prediction(pil_image)
                    current_results['ArcFace'] = {
                        'name': name,
                        'confidence': conf
                    }
                
                if use_custom and custom_model is not None:
                    name, conf, _ = get_custom_model_prediction(pil_image)
                    current_results['Custom Model'] = {
                        'name': name,
                        'confidence': conf
                    }
            
            # Afficher uniquement la meilleure prédiction pour chaque modèle
            for model_name, placeholder in pred_placeholders.items():
                if model_name in current_results:
                    data = current_results[model_name]
                    if data['confidence'] >= confidence_threshold:
                        color = "#4CAF50"  # Green
                        status = "✔️"
                    else:
                        color = "#FF9800"  # Orange
                        status = "⚠️"
                    
                    placeholder.markdown(
                        f"""
                        <div class="prediction-box">
                            <h4>{model_name} {status}</h4>
                            <p><strong>Person:</strong> {data['name']}</p>
                            <p><strong>Confidence:</strong> {data['confidence']:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        else:
            st.write('Webcam is stopped')
            camera.release()

if __name__ == "__main__":
    if facenet_model is None or arcface_app is None or custom_model is None:
        st.error("One or more models failed to load. Please check the model files.")
    else:
        main()