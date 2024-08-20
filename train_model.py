import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
from datetime import datetime

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data(data_dir):
    images = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(os.listdir(data_dir))}
    
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            images.append(image)
            labels.append(label_map[person_name])
    
    return np.array(images), np.array(labels), label_map

def build_custom_resnet50_model(input_shape=(224, 224, 3), num_classes=None):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Adding custom layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)  # Additional dense layer
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = True  # Unfreeze some layers for fine-tuning if needed
    
    return model

def build_custom_vggface_model(input_shape=(224, 224, 3), num_classes=None):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Adding custom layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)  # Additional dense layer
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = True  # Unfreeze some layers for fine-tuning if needed
    
    return model

def custom_cnn(input_shape=(224, 224, 3), num_classes=None):
    input_layer = Input(shape=input_shape)

    # First Block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Second Block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Third Block
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Fourth Block
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Fifth Block (Additional)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Additional Dense Layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output Layer
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

class DeleteOldCheckpoint(Callback):
    def __init__(self, checkpoint_dir):
        super(DeleteOldCheckpoint, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.previous_checkpoint = None

    def on_epoch_end(self, epoch, logs=None):
        # After each epoch, ModelCheckpoint will have saved the latest best model.
        # We check for the new model and delete the old one if it exists.
        if self.previous_checkpoint and os.path.exists(self.previous_checkpoint):
            os.remove(self.previous_checkpoint)
        self.previous_checkpoint = self.model_checkpoint_path()

    def model_checkpoint_path(self):
        # ModelCheckpoint saves the model using the epoch number.
        # Assuming the file format 'model_epoch*.h5', where * is the epoch number.
        # This will get the latest checkpoint file.
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.h5')]
        if checkpoint_files:
            latest_file = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(self.checkpoint_dir, f)))
            return os.path.join(self.checkpoint_dir, latest_file)
        return None

def train_strong_model(data_dir, model_dir):
    # Load data
    images, labels, label_map = load_data(data_dir)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    # Use the custom CNN model
    model = custom_cnn(input_shape=(224, 224, 3), num_classes=len(label_map))
    model_name = "custom_cnn"

    # Adjust the learning rate as suggested
    optimizer = SGD(learning_rate=0.0001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks for training
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_dir, 'model_epoch{epoch:02d}_val_loss{val_loss:.4f}.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    delete_old_checkpoint = DeleteOldCheckpoint(checkpoint_dir=model_dir)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    # Compute class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Train the model
    history = model.fit(datagen.flow(x_train, y_train, batch_size=4), 
                        epochs=200, 
                        validation_data=(x_test, y_test), 
                        class_weight=class_weights,
                        callbacks=[checkpoint, delete_old_checkpoint, early_stopping, reduce_lr])

    # The second phase of training (fine-tuning)
    model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_ft = model.fit(datagen.flow(x_train, y_train, batch_size=4), 
                           epochs=200, 
                           validation_data=(x_test, y_test), 
                           class_weight=class_weights,
                           callbacks=[checkpoint, delete_old_checkpoint, early_stopping, reduce_lr])
    
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"[INFO] Test accuracy: {test_accuracy:.4f}")

    # Optionally, save the label map and the final best model
    if delete_old_checkpoint.previous_checkpoint:
        final_model_path = delete_old_checkpoint.previous_checkpoint
        label_map_filename = f'{model_name}_label_map_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npy'
        np.save(os.path.join(model_dir, label_map_filename), label_map)
        print(f"[INFO] Model saved as {final_model_path}")
        print(f"[INFO] Label map saved as {label_map_filename}")

# if __name__ == "__main__":
#     data_dir = r'G:\Eravend\processed_data'
#     model_dir = r'G:\Eravend\\models'
#     train_strong_model(data_dir, model_dir)


