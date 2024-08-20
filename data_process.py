import os
import cv2
import numpy as np

data_dir = r'G:\Eravend\18_aug_face\data'
output_dir = r'G:\Eravend\18_aug_face\processed_data'

def preprocess_data(data_dir, output_dir):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("[ERROR] Failed to load Haar Cascade classifier")
        return
    
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        output_person_dir = os.path.join(output_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[ERROR] Failed to read image: {img_path}")
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                print(f"[WARNING] No face detected in image: {img_path}")
                continue
            
            # Assuming the largest face is the target
            x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
            face = image[y:y+h, x:x+w]  # Crop to face only
            face = cv2.resize(face, (224, 224))  # Resize to 224x224

            output_path = os.path.join(output_person_dir, img_name)
            cv2.imwrite(output_path, face)
            print(f"[INFO] Processed and saved: {output_path}")

# if __name__ == "__main__":
#     preprocess_data(data_dir, output_dir)


