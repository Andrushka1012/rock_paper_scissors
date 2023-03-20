from tensorflow import keras
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

from data.dataset import INPUT_WIDTH, INPUT_HEIGHT


class CnnModel(keras.Model):
    def __init__(self, num_classes=4):
        super().__init__()

        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_WIDTH, INPUT_HEIGHT, 3))
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.pool3 = MaxPooling2D((2, 2))
        self.conv4 = Conv2D(256, (3, 3), activation='relu')
        self.pool4 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        # self.dropout = Dropout(0.5)
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(256, activation='relu')
        self.d3 = Dense(num_classes, activation='softmax')

        self.build((None, INPUT_WIDTH, INPUT_HEIGHT, 3))
        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        # x = self.dropout(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x
