import os
import cv2

def capture_photos(person_name, data_dir):
    # Create a directory for the person if it doesn't exist
    person_dir = os.path.join(data_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access the webcam")
        return
    
    instructions = [
        "Neutral expression", 
        "Smile", 
        "Turn your face slightly to the left", 
        "Turn your face slightly to the right", 
        "Look up", 
        "Look down",
        "Close-up shot", 
        "Further away shot"
    ]
    
    photo_count = 0
    instruction_index = 0

    while True:
        # Get the current instruction
        if instruction_index < len(instructions):
            instruction = instructions[instruction_index]
            print(f"\n[INFO] {instruction}. Press 's' to capture the photo.")
        else:
            print(f"\n[INFO] Capture a photo with any pose you like. Press 's' to capture, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame from webcam")
                break
            
            cv2.imshow('Capture Photo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                photo_count += 1
                photo_path = os.path.join(person_dir, f'{person_name}_{photo_count}.jpg')
                cv2.imwrite(photo_path, frame)
                print(f"[INFO] Saved photo to: {photo_path}")
                
                instruction_index += 1
                break
            elif key == ord('q'):
                print("[INFO] Capture cancelled.")
                break

        if photo_count >= 20:
            print(f"[INFO] You've captured {photo_count} photos.")
            cont = input("Do you want to capture more photos? (y/n): ").strip().lower()
            if cont != 'y':
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {photo_count} photos saved for {person_name}")

# if __name__ == "__main__":
#     data_dir = r'G:\Eravend\data'
#     person_name = input("Enter the name of the person: ").strip()
#     capture_photos(person_name, data_dir)




