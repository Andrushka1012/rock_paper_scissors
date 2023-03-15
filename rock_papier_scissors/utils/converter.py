import tensorflow as tf


def convert(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('rock_paper_scissors_model.tflite', 'wb') as f:
        f.write(tflite_model)
