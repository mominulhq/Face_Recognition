import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

def recognize_face_live(model_dir, input_size=(224, 224)):
    # Load the trained model and label map
    model_path = os.path.join(model_dir, r'----------------')  # Update with the correct model filename
    label_map_path = os.path.join(model_dir, r'------------------------')  # Update with the correct label map filename
    
    model = load_model(model_path)
    label_map = np.load(label_map_path, allow_pickle=True).item()

    # Start video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access the webcam")
        return
    
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from webcam")
            break
        
        # Convert the frame to grayscale (used for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Extract the face ROI and resize it to match the model's input size
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, input_size)
            face = face / 255.0  # Normalize the image
            face = np.expand_dims(face, axis=0)  # Add batch dimension
            
            # Make a prediction on the face
            prediction = model.predict(face)
            predicted_label_index = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            # Determine the label based on confidence
            label = "Unverified" if confidence < 0.7 else list(label_map.keys())[list(label_map.values()).index(predicted_label_index)]
            label = f"{label} ({confidence:.2f})"
            
            # Display the label and bounding box on the frame
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (36,255,12), 2)
        
        # Show the processed video frame
        cv2.imshow('Live Face Recognition', frame)
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# def main():
#     model_dir = r'G:\Eravend\18_aug_face\models'  # Update this path as needed
#     recognize_face_live(model_dir)

# if __name__ == "__main__":
#     main()


















'''import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

def recognize_face_live(model_dir):
    model_path = os.path.join(model_dir, 'face_recognition_model_custom.h5')
    label_map_path = os.path.join(model_dir, 'label_map.npy')
    
    model = load_model(model_path)
    label_map = np.load(label_map_path, allow_pickle=True).item()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access the webcam")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from webcam")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (250, 250))
            face = face / 255.0
            face = face.reshape(1, 250, 250, 3)
            
            prediction = model.predict(face)
            predicted_label_index = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            label = "Unverified" if confidence < 0.7 else list(label_map.keys())[list(label_map.values()).index(predicted_label_index)]
            label = f"{label} ({confidence:.2f})"
            
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (36,255,12), 2)
        
        cv2.imshow('Live Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# No need for if __name__ == "__main__": here
'''