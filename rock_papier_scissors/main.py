from tkinter import Tk
from tkinter.filedialog import askopenfilename

import numpy as np
import cv2
from data.dataset import preprocess_image, get_merged_dataset, INPUT_HEIGHT, INPUT_WIDTH, generate_noice, \
    generate_nothing_dataset, CATEGORIES
from training.cnn import train_cnn

from training.vgg16 import train_vgg16, restore_vgg16
from utils.converter import convert


def predict_on_file(model):
    Tk().withdraw()
    img_path = askopenfilename()

    img = preprocess_image(img_path)
    result = model.predict(img)
    category_label = CATEGORIES[np.argmax(result)]
    print(f'Found: {category_label}')


def predict_on_video(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if ret:
            resized_frame = cv2.resize(frame, (INPUT_HEIGHT, INPUT_WIDTH))
            input_tensor = np.expand_dims(resized_frame, axis=0)
            # Predict the digit using the model
            pred = model.predict(input_tensor)[0]
            normalised_result = np.argmax(pred)
            category_label = CATEGORIES[normalised_result]

            text = f'Found: {category_label}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_trained():
    train_images_merged, train_labels_merged, test_images_merged, test_labels_merged = get_merged_dataset()

    train_images = train_images_merged
    train_labels = train_labels_merged

    model = restore_vgg16(test_model=True)
    return model


rpc_model = get_trained()
predict_on_video(rpc_model)

# Convert to tf format
# convert(rpc_model)

# # while True:
# #     predict_on_file(rpc_model)
