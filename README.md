# Hand Sign Detection using Computer Vision

## Overview
This project implements a real-time hand sign detection system using computer vision techniques, specifically targeting American Sign Language (ASL). The system detects hands through a webcam, preprocesses the images, and classifies hand signs into categories such as A, B, and C. It can be extended to include additional signs.

The pipeline integrates:
- **Hand detection** using MediaPipe via CVZone
- **Image preprocessing** with consistent sizing, centering, and aspect ratio management
- **Classification** using a pretrained TensorFlow/Keras model
- **Real-time visualization** with bounding boxes and labels

---

## Features
- 🚀 **End-to-End Pipeline:** Complete workflow from data collection to real-time classification.
- 📸 **Data Collection:** Capture images via webcam, save into category-specific folders.
- 📏 **Image Preprocessing:** Crop hand regions, resize to 300x300 pixels, preserve aspect ratio.
- 🧠 **Model Training and Export:** Pretrained Keras model integrated into Python code for live detection.
- 🖥️ **Real-Time Visualization:** Bounding boxes and labels overlayed on detected hands.
- ⚙️ **Keyboard Interaction:** Save images or perform actions using keystrokes.

---
