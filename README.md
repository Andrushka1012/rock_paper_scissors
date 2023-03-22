
This repository contains the code for a Flutter app that uses a custom Convolutional Neural Network to classify hand gestures as rock, paper, or scissors. The app was developed as part of a series of Medium articles "How We Built a Winning Model with Flutter and VGG16", which started with "Conquer Rock-Paper-Scissors with Flutter and a Custom VGG16 Model for Image Classification" (https://medium.com/p/a81d3f671b94).

The app uses the camera of the device to capture an image of the user's hand gesture, and then uses our image classification model to predict the correct gesture. The project is an example of how to integrate a custom machine learning model into a Flutter app, making use of the platform's built-in features such as the camera.

## Installation

To run this app, you will need to have Flutter installed on your machine. Once Flutter is installed, clone this repository and run `flutter run` on rock_paper_scissors_mobile to start the app.

## Usage

When you open the app, you will see a camera viewfinder. Place your hand in front of the camera and make a gesture of either rock, paper, or scissors. The app will capture an image of your hand gesture and use our image classification model to predict the correct gesture. The predicted gesture will be displayed on the screen.

## Development

This app was developed using the Flutter framework and the Keras library for Python. The custom Convolutional Neural Network was trained using a small dataset of hand gesture images.

It uses transfer learning based on pretrained VGG-16 model to achive better performance. 

If you would like to modify or improve the image classification model, you can do so by editing the `vgg16_rock_paper_scissors_model.py` file and retraining the model. Once you have a new model, you can replace the `rock_paper_scissors_model.tflite` file in the `assets` folder with the new model file.

## Credits

This project was inspired by the Rock-Paper-Scissors image classification task on Kaggle ([https://www.kaggle.com/drgfreeman/rockpaperscissors](https://www.kaggle.com/drgfreeman/rockpaperscissors)). The dataset used to train the image classification model was sourced from this Kaggle task.

The Flutter app was developed by Andrii Makarenko (https://www.linkedin.com/in/andrii-makarenko/) and the image classification model was trained by Andrii Makarenko (https://www.linkedin.com/in/andrii-makarenko/) .

## License

This project is licensed under the MIT License
