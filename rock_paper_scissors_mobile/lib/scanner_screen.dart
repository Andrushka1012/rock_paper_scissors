import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:rock_paper_scissors_mobile/clasifier.dart';
import 'package:rock_paper_scissors_mobile/image_utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class ScannerScreen extends StatefulWidget {
  @override
  _ScannerScreenState createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  late CameraController cameraController;
  late Interpreter interpreter;
  final classifier = Classifier();

  bool isWorking = false;
  bool initialized = false;
  DateTime lastShot = DateTime.now();

  @override
  void initState() {
    super.initState();
    initialize();
  }

  Future<void> initialize() async {
    // Load the TensorFlow Lite model
    await loadModel();

    await classifier.loadModel();

    final cameras = await availableCameras();
    // Create a CameraController object
    cameraController = CameraController(
      cameras[0], // Choose the first camera in the list
      ResolutionPreset.medium, // Choose a resolution preset
    );

    // Initialize the CameraController and start the camera preview
    await cameraController.initialize();
    await cameraController.startImageStream((image) {
      if (DateTime
          .now()
          .difference(lastShot)
          .inSeconds > 3) {
        isWorking = true;
        processCameraImage(image);
      }
    });

    setState(() {
      initialized = true;
    });
  }

  Future<void> loadModel() async {
    // Load the TensorFlow Lite model
    try {
      interpreter = await Interpreter.fromAsset('rock_paper_scissors_model.tflite');
      final inputTensor = interpreter.getInputTensor(0);
      print("Input shape ${inputTensor.shape}");
    } catch (e) {
      print(e);
    }
  }

  Future<void> processCameraImage(CameraImage cameraImage) async {
    print('porccessing');
    final convertedImage = ImageUtils.convertYUV420ToImage(cameraImage);

    await classifier.predict(convertedImage);

    isWorking = false;
    lastShot = DateTime.now();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter Camera Demo'),
      ),
      body: initialized ? CameraPreview(cameraController) : const Center(child: CircularProgressIndicator()),
    );
  }

  @override
  void dispose() {
    cameraController.dispose();
    super.dispose();
  }
}
