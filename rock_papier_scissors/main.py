from tkinter import Tk
from tkinter.filedialog import askopenfilename

import numpy as np
from data.dataset import preprocess_image, get_merged_dataset

from training.vgg16 import train_vgg16

train_images_merged, train_labels_merged, test_images_merged, test_labels_merged = get_merged_dataset()

train_images = train_images_merged
train_labels = train_labels_merged


def predict_on_file(model):
    Tk().withdraw()
    img_path = askopenfilename()

    img = preprocess_image(img_path)
    result = model.predict(img)
    print(f'result: {np.argmax(result)}')


vgg_rps_model = train_vgg16(train_images, train_labels)

while True:
    predict_on_file(vgg_rps_model)
