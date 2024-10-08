# Face Recognition System for EraVEnd

This repository hosts the development of a sophisticated face recognition system created for EraVEnd, a German company specializing in cutting-edge security and identification solutions. The system leverages deep learning frameworks such as TensorFlow and Keras to build, train, and deploy models for accurate and efficient face recognition.

## Features

- **Custom Data Processing**: Includes scripts for capturing and processing image data tailored to specific individuals or groups.
- **Flexible Model Training**: Supports training of custom CNN, ResNet50, and VGG16-based models with fine-tuning capabilities.
- **High Accuracy Recognition**: Achieves high accuracy in face recognition tasks through state-of-the-art deep learning techniques.
- **Scalable and Modular**: The system is designed to be scalable, allowing for easy integration and extension for various applications.



```plaintext
face-recognition-system/
├── data/
│ ├── person1/
│ └── person2/
├── models/
│ ├── custom_cnn_model_date_time.h5
│ ├── custom_cnn_label_map_date_time.npy
│ ├── custom_resnet50_model_date_time.h5
│ ├── custom_resnet50_label_map_date_time.npy
│ ├── model_date_time.h5
│ └── label_map_date_time.npy
├── processed_data/
│ ├── person1/
│ └── person2/
├── capture_photos.py
├── data_process.py
├── How_to_use.md
├── recognize_face.py
├── train_model.py
└── main.py
```






## About EraVEnd

EraVEnd is a company based in Germany, specializing in innovative solutions for security, automation, and user identification. This face recognition system is part of EraVEnd's initiative to enhance security protocols and streamline user verification processes.
