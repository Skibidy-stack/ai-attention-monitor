# AI Eye Detector

## Overview
This project detects human eyes in real time using a webcam.  
It uses a Haar Cascade classifier to identify eye regions in each video frame.

The program captures frames from the webcam, processes them using OpenCV, and draws rectangles around detected eyes.

## Technologies Used
- Python
- OpenCV
- NumPy

## How It Works

1. Capture frame from webcam
2. Convert frame to grayscale
3. Apply Haar Cascade eye detector
4. Detect eye regions
5. Draw rectangles around detected eyes
6. Display the processed frame

## Installation

Install required libraries:

pip install opencv-python numpy

## Run the Program

python eye_detector.py

Press **q** to close the webcam window.

## Project Structure

ai-eye-detector  
│  
├── eye_detector.py  
└── README.md  

## Future Improvements
- Add face detection
- Implement blink detection
- Use deep learning models
- Deploy as a web application