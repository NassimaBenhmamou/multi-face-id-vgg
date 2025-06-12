import cv2
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN

# Création d'une instance globale MTCNN (detector de visages)
mtcnn = MTCNN(keep_all=True)

class WebcamCapture:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Impossible d'accéder à la webcam.")
    
    def get_frame_with_boxes(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Conversion BGR (OpenCV) -> RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Détection des visages avec MTCNN
        boxes, _ = mtcnn.detect(image)

        # Dessiner les boxes si détectées
        if boxes is not None:
            draw = ImageDraw.Draw(image)
            for box in boxes:
                # box est un array [x1, y1, x2, y2]
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)

        return image
    
    def release(self):
        self.cap.release()

# Fonction simple pour capture ponctuelle (comme avant)
def capture_realtime_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Erreur : Impossible d’accéder à la webcam.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Erreur : Échec de la capture d'image.")
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    return image
