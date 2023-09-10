```markdown
# Real-Time Face Detection with OpenVINO and OpenCV

This repository hosts a Python script that leverages the OpenVINO toolkit and the OpenCV library to facilitate real-time face detection using a webcam.

## Prerequisites

- OpenVINO Toolkit
- OpenCV Python Package

## Setting Up

1. **OpenVINO Toolkit:** Follow the detailed setup instructions on the [official website](https://docs.openvinotoolkit.org/latest/index.html).
2. **OpenCV Python:** Install it using the following pip command:
   ```
   pip install opencv-python-headless
   ```

## How to Use

1. Clone the repository:
   ```
   git clone https://github.com/milaomrani/openvino_face-detection.git
   ```
   
2. Change to the project directory:
   ```
   cd openvino_face-detection
   ```

3. Execute the script:
   ```
   python main.py
   ```

### Important Points:
- The script operates using the 'face-detection-adas-0001' model. Ensure the '.xml' and '.bin' files are located in your project directory.
- To close the webcam window, press the 'q' key.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Support

If you encounter any problems or have questions, feel free to open an issue on the GitHub repository.
```
