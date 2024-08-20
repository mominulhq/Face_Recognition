import argparse
from capture_photos import capture_photos
from data_process import preprocess_data
from train_model import train_strong_model
from recognize_face import recognize_face_live

def main():
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument('--mode', type=str, required=True, choices=['capture', 'preprocess', 'train', 'recognize'],
                        help='Mode of operation: capture, preprocess, train, recognize')

    args = parser.parse_args()

    if args.mode == 'capture':
        person_name = input("Enter the name of the person: ").strip()
        data_dir = './data'  # Update this path as needed
        capture_photos(person_name, data_dir)

    elif args.mode == 'preprocess':
        data_dir = './data'  # Update this path as needed
        output_dir = './processed_data'  # Update this path as needed
        preprocess_data(data_dir, output_dir)

    elif args.mode == 'train':
        data_dir = './processed_data'  # Update this path as needed
        model_dir = './models'  # Update this path as needed
        train_strong_model(data_dir, model_dir)

    elif args.mode == 'recognize':
        model_dir = './models'  # Update this path as needed
        recognize_face_live(model_dir)

if __name__ == "__main__":
    main()
