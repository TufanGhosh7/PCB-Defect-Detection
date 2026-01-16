# PCB-Defect-Detection
PCB Defect Detection using Machine Learning

This project presents an automated PCB Defect Detection system that combines Differential Image Processing techniques with CNN-based Deep Learning models to accurately detect and classify defects in Printed Circuit Boards (PCBs).
A Streamlit-based web application is developed to provide real-time inference with visual defect localization.

The system aims to reduce manual inspection effort, improve defect detection accuracy, and enhance quality control in electronics manufacturing.

---

## Problem Statement
Manual inspection of Printed Circuit Boards is time-consuming, inconsistent, and prone to human error, especially in large-scale manufacturing.
Traditional rule-based Automated Optical Inspection (AOI) systems struggle with variations in lighting, alignment, and complex defect patterns.

---

## Objectives
- Detect PCB defects using differential image processing
- Localize defect regions using SSIM and sliding window analysis
- Classify defects using CNN-based deep learning models
- Compare performance of multiple pretrained CNN architectures
- Visualize results using confusion matrix and classification reports
- Deploy the best-performing model using a Streamlit web interface

---

## Technologies Used
Programming Language: Python
Image Processing: OpenCV, NumPy, scikit-image
Deep Learning: PyTorch / TensorFlow
CNN Architectures: ResNet18, ResNet50
Visualization: Matplotlib
Web Application: Streamlit
Platform: Google Colab, GitHub

---

## Dataset

The project uses a PCB defect dataset consisting of golden reference PCB images and test PCB images with various defects.

## Defect Classes
- Missing Hole
- Mouse Bite
- Open Circuit
- Short Circuit
- Spur
- Spurious Copper

---

- **Visualization:** Matplotlib, Seaborn  

---

## Image Processing Techniques Used

- Image resizing
- Grayscale conversion
- Noise removal
- Normalization
- Data augmentation

---

## Deep Learning Models Used

- Custom CNN
- Pretrained CNN models (ResNet18 / Resnet50)

## Model Evaluation
The models are evaluated using:
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- Training and Validation Accuracy
- Training and Validation Loss
- 
The best-performing model (Resnet50) achieved an accuracy of approximately 99.7%

---

## Streamlit Web Application
A Streamlit-based user interface is developed for real-time PCB defect detection.

## Features
- Upload PCB image
- Automatic defect detection and classification
- Bounding boxes with defect labels
- Download result image and log file

# Run Streamlit App

```bash
cd PCB_Dataset/streamlit_inference
pip install -r requirements.txt
streamlit run app.py
```

## Results
- Differential image processing effectively localizes defect regions
- CNN models accurately classify PCB defect types
- ResNet50 provides high accuracy and strong generalization
- Streamlit interface enables easy and interactive usage

---

## Limitations
- Requires availability of a golden reference PCB image
- Sliding window approach increases inference time
- Dataset contains limited PCB layouts
- Performance may vary under extreme lighting conditions

## Future Scope
Integration of object detection models (YOLO, Faster R-CNN)
Real-time PCB inspection using industrial cameras
Expansion to larger and more diverse PCB datasets
Deployment on edge devices and industrial hardware

---

## Conclusion
This project demonstrates a robust and efficient PCB defect detection system by combining differential image processing with CNN-based deep learning models.
The hybrid approach improves defect localization and classification accuracy, while the Streamlit web application provides a user-friendly interface for real-time inspection.






