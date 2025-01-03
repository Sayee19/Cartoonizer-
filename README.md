# Cartoon
I created a cartoonizer (2D image animator) by referencing the Image to Cartoon project on [ https://projectworlds.in/image-to-cartoon-python-opencv-machine-learning/ ], utilizing Python, OpenCV, and machine learning techniques for image transformation.

## Installation

### Application tested on:

- python 3.7
- tensorflow 2.1.0 
- tf_slim 1.1.0
- ffmpeg 3.4.8
- Cuda version 10.1
- OS: Linux (Ubuntu 18.04)


### Using `virtualenv`

1. Make a virtual environment using `virutalenv` and activate it
```
virtualenv -p python3 cartoonize
source cartoonize/bin/activate
```
2. Install python dependencies
```
pip install -r requirements.txt
```
3. Run the webapp. Be sure to set the appropriate values in `config.yaml` file before running the application.
```
python app.py
```
