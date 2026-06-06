# Live Age and Gender Recognition

A Python program that uses a webcam to detect a face in real time and estimate the person's gender and age. It draws the result above each face, e.g. "Male (25-32)".

## How it works
1. Frames are captured from the webcam one by one (OpenCV).
2. Faces are detected in each frame (Haar cascade).
3. Each face is cropped and prepared (227x227 pixels, color-balanced).
4. Two pre-trained neural networks estimate gender and age.
5. The result is drawn on top of the video.

## Technologies used
- Python
- OpenCV (opencv-python)
- NumPy
- Pre-trained Caffe models for age and gender

## Models
The pre-trained models (age_net.caffemodel, age_deploy.prototxt, gender_net.caffemodel, gender_deploy.prototxt) were taken from this repository:

https://github.com/smahesh29/Gender-and-Age-Detection

The original age and gender models were created by Gil Levi and Tal Hassner, trained on the Adience dataset.

Adience dataset:

https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification

Note: the model files (*.caffemodel) are not included in this repository due to their size. Download them from the link above and place them in the same folder as the code.

## How to run
1. Install the libraries: **pip install opencv-python numpy**
2. 2. Download the 4 model files (see above) into the project folder.
3. Run: **python pol_uzrast.py**
4. Press Q to quit.

## Roadmap
Currently the project uses pre-trained models. The next step is to train a custom model on the Adience dataset (Google Colab) and compare accuracy.
