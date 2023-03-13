import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from model.cnn_model import CnnModel
from training.testing import test
from utils.callbacks import StopByAccuracyCallback
from utils.data_generalisation import get_generators

MODEL_NAME = 'cnn_rock_papier_scissors.h5'


def train_cnn(train_images, train_labels) -> Model:
    model = CnnModel()
    model.summary()

    train_generator, validation_generator = get_generators(train_images, train_labels)

    model.fit(
        train_generator,
        steps_per_epoch=40,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=10,
        callbacks=[StopByAccuracyCallback()]
    )
    model.save_weights(MODEL_NAME)

    test(model)

    return model


def restore_cnn(test_model=False) -> Model:
    model = CnnModel()

    model.load_weights(MODEL_NAME)

    if test_model:
        test(model)

    return model
