import os
from random import shuffle

import numpy as np
from keras.utils import load_img, img_to_array, to_categorical

INPUT_WIDTH = 100
INPUT_HEIGHT = 150

TARGET_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)
NUM_CATEGORIES = 3

TRAIN_DIR_1 = "data/Dataset/train"
TEST_DIR_1 = "data/Dataset/test/"
TRAIN_DIR_2 = "data/Dataset2/train"
TEST_DIR_2 = "data/Dataset2/test/"
TRAIN_DIR_3 = "data/Dataset3/train"
TEST_DIR_3 = "data/Dataset3/test/"

CATEGORIES = ["rock", "paper", "scissors"]


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
    return get_dataset(TRAIN_DIR_1, TEST_DIR_1)


def get_dataset_2():
    return get_dataset(TRAIN_DIR_2, TEST_DIR_2)


def get_dataset_3():
    return get_dataset(TRAIN_DIR_3, TEST_DIR_3)


def get_merged_dataset():
    train_images_1, train_labels_1, test_images_1, test_labels_1 = get_dataset_1()
    train_images_2, train_labels_2, test_images_2, test_labels_2 = get_dataset_2()
    train_images_3, train_labels_3, test_images_3, test_labels_3 = get_dataset_3()

    train_images = np.concatenate((train_images_1, train_images_2, train_images_3), axis=0)
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3), axis=0)
    test_images = np.concatenate((test_images_1, test_images_2, test_images_3), axis=0)
    test_labels = np.concatenate((test_labels_1, test_labels_2, test_labels_3), axis=0)

    return train_images, train_labels, test_images, test_labels
