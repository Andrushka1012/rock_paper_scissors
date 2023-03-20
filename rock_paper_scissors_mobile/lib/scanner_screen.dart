import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:rock_paper_scissors_mobile/clasifier.dart';
import 'package:rock_paper_scissors_mobile/image_utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'classes.dart';

class ScannerScreen extends StatefulWidget {
  @override
  _ScannerScreenState createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  late CameraController cameraController;
  late Interpreter interpreter;
  final classifier = Classifier();

  bool initialized = false;
  DetectionClasses detected = DetectionClasses.nothing;
  DateTime lastShot = DateTime.now();

  @override
  void initState() {
    super.initState();
    initialize();
  }

  Future<void> initialize() async {
    await classifier.loadModel();

    final cameras = await availableCameras();
    // Create a CameraController object
    cameraController = CameraController(
      cameras[0], // Choose the first camera in the list
      ResolutionPreset.medium, // Choose a resolution preset
    );

    // Initialize the CameraController and start the camera preview
    await cameraController.initialize();
    // Listen for image frames
    await cameraController.startImageStream((image) {
      // Make predictions every 1 second to avoid overloading the device
      if (DateTime.now().difference(lastShot).inSeconds > 1) {
        processCameraImage(image);
      }
    });

    setState(() {
      initialized = true;
    });
  }

  Future<void> processCameraImage(CameraImage cameraImage) async {
    final convertedImage = ImageUtils.convertYUV420ToImage(cameraImage);

    final result = await classifier.predict(convertedImage);

    if (detected != result) {
      setState(() {
        detected = result;
      });
    }

    lastShot = DateTime.now();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter Camera Demo'),
      ),
      body: initialized
          ? Column(
              children: [
                SizedBox(
                  height: MediaQuery.of(context).size.width,
                  width: MediaQuery.of(context).size.width,
                  child: CameraPreview(cameraController),
                ),
                Text(
                  "Detected: ${detected.label}",
                  style: const TextStyle(
                    fontSize: 28,
                    color: Colors.blue,
                  ),
                ),
              ],
            )
          : const Center(child: CircularProgressIndicator()),
    );
  }

  @override
  void dispose() {
    cameraController.dispose();
    super.dispose();
  }
}
