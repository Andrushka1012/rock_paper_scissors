from data.dataset import get_merged_dataset, get_dataset_1, get_dataset_2, get_dataset_3


def test(model):
    train_images_merged, train_labels_merged, test_images_merged, test_labels_merged = get_merged_dataset()
    train_images_1, train_labels_1, test_images_1, test_labels_1 = get_dataset_1()
    train_images_2, train_labels_2, test_images_2, test_labels_2 = get_dataset_2()
    train_images_3, train_labels_3, test_images_3, test_labels_3 = get_dataset_3()

    print('Test dataset 1')
    loss, accuracy = model.evaluate(test_images_1, test_labels_1)
    # Print the test results
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    print('Test dataset 2')
    loss, accuracy = model.evaluate(test_images_2, test_labels_2)
    # Print the test results
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    print('Test dataset 3')
    loss, accuracy = model.evaluate(test_images_3, test_labels_3)
    # Print the test results
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    print('Test merged dataset')
    loss, accuracy = model.evaluate(test_images_merged, test_labels_merged)
    # Print the test results
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
