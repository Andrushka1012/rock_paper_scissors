from keras.models import Model

from data.dataset import INPUT_WIDTH, INPUT_HEIGHT

import ssl

from model.vgg16_rock_paper_scissors_model import RockPaperScissorsVgg16
from training.testing import test
from utils.callbacks import StopByAccuracyCallback
from utils.data_generalisation import get_generators

ssl._create_default_https_context = ssl._create_unverified_context

MODEL_NAME = 'vgg16_rock_papier_scissors.h5'


def train_vgg16(train_images, train_labels) -> Model:
    vgg16 = RockPaperScissorsVgg16(INPUT_WIDTH, INPUT_HEIGHT)
    model = vgg16.model
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
    print(f"Save weights into {MODEL_NAME}")
    model.save_weights(MODEL_NAME)

    test(model)

    return model


def restore_vgg16(test_model=False) -> Model:
    vgg16 = RockPaperScissorsVgg16(INPUT_WIDTH, INPUT_HEIGHT)
    model = vgg16.model

    model.load_weights(MODEL_NAME)

    if test_model:
        test(model)

    return model
