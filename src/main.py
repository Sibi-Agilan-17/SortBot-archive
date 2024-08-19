import logging
import random
import sys

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

data_dir = "./dataset-resized"
MODEL = "trash_recognition_model_epoch100.h5"


def load_model(filename: str = MODEL) -> tf.keras.models.Model:
    """Load the model from the file if it exists, otherwise return a new model."""

    try:
        model = tf.keras.models.load_model(filename)
        model.summary()
        return model

    except (FileNotFoundError, OSError) as exc:
        logging.error(f'Error loading model: {exc}')
        sys.exit(-1)


def generate_dataset(subset: str = "training", split: float = 0.1, batch_size: int = 16) -> tf.data.Dataset:
    """Generate a dataset from the directory."""

    return image_dataset_from_directory(
        data_dir,
        validation_split=split,
        subset=subset,
        seed=random.randint(2 ** 16, 2 ** 32 - 1),
        image_size=(256, 256),  # resize if needed
        batch_size=batch_size
    )


def test_model(times: int = 100, split: float = 0.1) -> float:
    """Test the model on the dataset and return the average accuracy."""

    avg_acc = 0

    model = load_model()
    test_dataset = generate_dataset(subset="validation", split=split)

    for _ in range(times):
        _, test_acc = model.evaluate(test_dataset)
        avg_acc = (avg_acc + test_acc) / 2

    print(f'Average accuracy: {avg_acc}')
    logging.info(f'{times} tested; average accuracy: {avg_acc}')

    return avg_acc


# Call the test_model function
test_model()
