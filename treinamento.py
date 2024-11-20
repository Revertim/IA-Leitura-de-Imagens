import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_and_save_model():

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = 'archive/data/data/train'
    val_dir = 'archive/data/data/val'

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        batch_size=32,
        class_mode='sparse'
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(28, 28),
        batch_size=32,
        class_mode='sparse'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(36, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10
    )

    model.save('model.h5')
    print("Modelo salvo como 'model.h5'")

train_and_save_model()
