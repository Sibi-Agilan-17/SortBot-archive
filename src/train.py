import logging
import random
import sys

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D

data_dir: str = "./dataset-resized"  # or use `./dataset-original` if not resized


# Custom formatter to colorize error messages
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.ERROR:
            record.msg = f"\033[91m{record.msg}\033[0m"  # Red color
        return super().format(record)


# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# StreamHandler for terminal output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(CustomFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# FileHandler for writing info and debug logs to a file
file_handler = logging.FileHandler('info_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Define the model
new_model = tf.keras.Sequential()

# First Convolutional Block
new_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
new_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Block
new_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
new_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Block
new_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
new_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Fourth Convolutional Block
new_model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
new_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten layer to transition to Dense layers
new_model.add(tf.keras.layers.Flatten())

# Fully connected Dense Layer
new_model.add(tf.keras.layers.Dense(128, activation='relu'))
new_model.add(tf.keras.layers.Dropout(0.5))  # Dropout for regularization

# Output Layer
new_model.add(tf.keras.layers.Dense(6, activation='softmax'))  # 6 output categories


def load_model(model_path: str) -> tf.keras.models.Model:
    """Load the model from the file if it exists, otherwise return a new model."""
    try:
        # load the model
        model = tf.keras.models.load_model(model_path)
    except FileNotFoundError or OSError as exc:
        model = new_model
        logging.error(f'Error loading model: {exc}')

    model.summary()
    return model


def generate_dataset(subset: str = "training", batch_size: int = 16) -> tf.data.Dataset:
    """Generate a dataset from the directory."""
    return image_dataset_from_directory(
        data_dir,
        validation_split=0.01,
        subset=subset,
        seed=random.randint(2 ** 16, 2 ** 32 - 1),
        image_size=(256, 256),  # resize if needed
        batch_size=batch_size
    )


def test_model(model_path: str, times: int = 10) -> int:
    """Test the model on the dataset."""
    avg_acc = 0

    model = load_model(model_path)
    test_dataset = generate_dataset(subset="validation")

    for _ in range(times):
        _, test_acc = model.evaluate(test_dataset)
        avg_acc = (avg_acc + test_acc) / 2

    print(f'Average accuracy: {avg_acc}')
    logging.info(f'{times} tested; average accuracy: {avg_acc}')
    return avg_acc


def train_model(model_path: str, epochs: int = 10, test: bool = True) -> tf.keras.callbacks.History:
    """Train the model on the dataset."""
    model = load_model(model_path)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],)
    logging.debug('Model compiled')

    dataset = generate_dataset()
    validation_dataset = generate_dataset(subset="validation")
    logging.debug('Dataset loaded')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    train_history = model.fit(
        dataset,
        validation_data=validation_dataset,
        epochs=epochs,  # Number of epochs for training
        callbacks=[early_stopping],
    )

    if test:
        test_model(model_path)

    # Save the model
    model.save(model_path.replace('.h5', '.keras'))
    logging.debug('Model saved')

    return train_history


def plot_history(history: tf.keras.callbacks.History):
    """Plot the training and validation history."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    history = train_model("sortbot_beta.h5", epochs=25)
    plot_history(history)
