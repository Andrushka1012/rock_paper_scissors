from keras.preprocessing.image import ImageDataGenerator


def get_generators(train_images, train_labels, batch_size=32):
    # Define the augmentation parameters
    # Apply the augmentation to the training data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        horizontal_flip=True,
        shear_range=0.2,
        fill_mode='wrap',
        validation_split=0.2
    )

    # Apply the augmentation to the training data
    train_generator = train_datagen.flow(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        subset='training'
    )

    validation_generator = train_datagen.flow(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        subset='validation'
    )

    return train_generator, validation_generator
