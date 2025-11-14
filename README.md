# Color\_Detection 🎨

---

## Description

This repository hosts a very simple **AICV (Artificial Intelligence/Computer Vision) project** that utilizes fundamental **image processing** techniques to detect specific colors within an image or video stream.

It serves as a straightforward example for beginners looking to understand how to apply color masking and contour detection using Python.

---

## Features

* **Real-time Color Detection:** Process live video feed from a webcam (implied by typical CV projects).
* **Image Processing:** Apply techniques like converting to **HSV color space** for accurate color segmentation.
* **Object Isolation:** Use **color masks** to isolate and identify objects of a specific color.
* **Boundary Tracing:** Draw bounding boxes or contours around the detected color regions.

---

## Prerequisites

To run this project, you will need **Python 3.x** installed on your system.

### Dependencies

This project relies on common Python computer vision libraries. You can install all necessary dependencies using `pip`:

Bash

```
pip install opencv-python numpy

```

| Dependency          | Description                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| opencv-python (cv2) | The primary library for computer vision tasks, used for image manipulation, color space conversion, and video capture. |
| numpy               | Used for efficient array manipulation, especially when handling image data.                                            |


## Installation

Follow these steps to get a local copy up and running.

1. **Clone the repository:**  
Bash  
```  
git clone https://github.com/olliewh00/Color_Detection.git  
```
2. **Navigate to the project directory:**  
Bash  
```  
cd Color_Detection  
```
3. **Install dependencies** (if you haven't already):  
Bash  
```  
pip install -r requirements.txt  
# OR: pip install opencv-python numpy  
```  
_Note: It's recommended to run this project inside a Python [virtual environment](https://docs.python.org/3/library/venv.html)._

---

## Usage

The main detection logic is contained within the `main.py` file.

To start the color detection program (assuming it's set up to use your webcam), simply run:

Bash

```
python main.py

```

### ⚙️ Configuration (Common Adjustments)

Color detection often requires fine-tuning the **HSV (Hue, Saturation, Value)** color range for the specific color you want to detect.

If you are detecting a color like **Blue**, you would typically find the HSV range defined in `main.py`:

Python

```
# Example of a typical Blue color range in HSV (check main.py for actual values)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

```

You may need to modify these `lower_` and `upper_` bounds in the `main.py` file to suit different lighting conditions or shades of the target color.

---


