import os
import uuid
from io import BytesIO
from random import shuffle

import numpy as np
from PIL import Image
from keras.utils import load_img, img_to_array, to_categorical
import requests
from PIL import Image

INPUT_WIDTH = 150
INPUT_HEIGHT = 150

TARGET_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)
NUM_CATEGORIES = 4

TRAIN_DIR_1 = "data/Dataset/train"
TEST_DIR_1 = "data/Dataset/test"
TRAIN_DIR_2 = "data/Dataset2/train"
TEST_DIR_2 = "data/Dataset2/test"
TRAIN_DIR_3 = "data/Dataset3/train"
TEST_DIR_3 = "data/Dataset3/test"

CATEGORIES = ["rock", "paper", "scissors", "nothing"]


def fetch_images(cat, directory):
    images_path = [f"{directory}/{cat}/{f}" for f in os.listdir(f"{directory}/{cat}") if
                   f.endswith(".jpeg") or f.endswith(".jpg")]
    preprocessed = [preprocess_image(x) for x in images_path]
    return preprocessed


def preprocess_image(image_path):
    img = load_img(image_path, target_size=TARGET_SIZE)
    img = img.convert("RGB")

    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def get_dataset(train, test):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for i, category in enumerate(CATEGORIES):
        train_images_cat = fetch_images(category, train)
        test_images_cat = fetch_images(category, test)

        train_labels_cat = to_categorical(np.full(len(train_images_cat), i), NUM_CATEGORIES)
        test_labels_cat = to_categorical(np.full(len(test_images_cat), i), NUM_CATEGORIES)

        train_images.extend(train_images_cat)
        test_images.extend(test_images_cat)
        train_labels.extend(train_labels_cat)
        test_labels.extend(test_labels_cat)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    train_images = np.squeeze(train_images, axis=1)
    test_images = np.squeeze(test_images, axis=1)
    # shuffle elements in train dataset
    combined = list(zip(train_images, train_labels))
    shuffle(combined)
    train_images, train_labels = zip(*combined)

    return np.array(train_images), np.array(train_labels), test_images, test_labels


def get_dataset_1():
    print("Load dataset_1")
    return get_dataset(TRAIN_DIR_1, TEST_DIR_1)


def get_dataset_2():
    print("Load dataset_2")
    return get_dataset(TRAIN_DIR_2, TEST_DIR_2)


def get_dataset_3():
    print("Load dataset_3")
    return get_dataset(TRAIN_DIR_3, TEST_DIR_3)


def get_merged_dataset():
    print("Load merged dataset")
    train_images_1, train_labels_1, test_images_1, test_labels_1 = get_dataset_1()
    train_images_2, train_labels_2, test_images_2, test_labels_2 = get_dataset_2()
    train_images_3, train_labels_3, test_images_3, test_labels_3 = get_dataset_3()

    train_images = np.concatenate((train_images_1, train_images_2, train_images_3), axis=0)
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3), axis=0)
    test_images = np.concatenate((test_images_1, test_images_2, test_images_3), axis=0)
    test_labels = np.concatenate((test_labels_1, test_labels_2, test_labels_3), axis=0)

    return train_images, train_labels, test_images, test_labels


def generate_noice(original_dir: str):
    url = f"https://picsum.photos/{INPUT_WIDTH}/{INPUT_HEIGHT}"

    # Send a GET request to the URL and receive the response
    response = requests.get(url)

    # Open the response content as an image using PIL
    image = Image.open(BytesIO(response.content))

    # Save the image to your local machine
    image.save(f"{original_dir}/nothing/{str(uuid.uuid4())}.jpg")


# Generate nothing dataset with random images
def generate_nothing_dataset():
    for i in range(100):
        generate_noice(TRAIN_DIR_1)
    for i in range(30):
        generate_noice(TEST_DIR_1)

    for i in range(500):
        generate_noice(TRAIN_DIR_2)
    for i in range(125):
        generate_noice(TEST_DIR_2)

    for i in range(400):
        generate_noice(TRAIN_DIR_3)
    for i in range(30):
        generate_noice(TEST_DIR_3)
