Hi!

This project harnesses the capabilities of OpenCV and the YOLOv8 model via the Hugging Face Hub to detect faces in real-time from a webcam feed. 
It's designed to be an accessible introduction to real-time object detection.

### Recommended Setup
I recommend using conda and pip for package management to ensure that all dependencies are handled smoothly. Visual Studio Code or PyCharm are excellent IDE choices for running and editing these scripts efficiently. 
Running complex real-time video processing scripts like this one in a Jupyter Notebook might not be ideal and can lead to unexpected issues due to pening the webcam in Spyder which often leads to python crashing.

The following libraries are needed for the project:

- numpy (for handling high-level mathematical operations and array manipulation)
- opencv-python (cv2, core library for real-time computer vision operations, especially for capturing and displaying video feeds)
- huggingface_hub (easy downloading and loading of models hosted on Hugging Face's Model Hub, in this case, the YOLOv8 model)
- ultralytics (facilitates model loading and inference with YOLO)
- Pillow (PIL.Image, used for image processing tasks not covered by OpenCV, and for displaying images)
- dotenv with load_dotenv (to use environment variables and hide sensible information)

Given the nature of face detection, it's crucial to remain compliant with local data protection laws. 
This script can be adapted for a variety of applications, but always consider the ethical implications and legal requirements, especially for surveillance purposes.

The script is annotated with comments to explain each line of code.

Have fun!
