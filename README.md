# Traffic Sign Recognition System

## Overview
This project implements a Deep Learning system for classifying traffic signs using a Convolutional Neural Network (CNN). The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which contains 43 different classes of traffic signs.

The goal is to provide a robust classification system that can be applied to Autonomous Vehicle (AV) software or Advanced Driver Assistance Systems (ADAS).

## Key Features
* **Architecture**: Custom 3-block Convolutional Neural Network built with PyTorch.
* **Data Augmentation**: Includes random rotation and color jittering to improve model generalization.
* **Training Pipeline**: Implements Early Stopping and Learning Rate Scheduling (ReduceLROnPlateau).
* **Visualization**: Real-time learning curves and a comprehensive Confusion Matrix for the most frequent classes.

## Technologies Used
* **Python**
* **PyTorch** (Deep Learning Framework)
* **Torchvision** (Computer Vision Utilities)
* **Matplotlib & Seaborn** (Data Visualization)
* **Scikit-learn** (Metrics and Evaluation)

## Dataset
The project uses the [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html), which consists of over 50,000 images of traffic signs across 43 categories. Images are resized to 32x32 pixels for processing.

## Performance & Results
The model achieves high accuracy on the validation set. Below are the key performance indicators:

### Learning Curves
![Learning Curves](learning_curves.png)
*Note: Run the script to generate and save your learning curves plot.*

### Sample Predictions
![Predictions](predictions.png)
*Note: Run the script to see sample predictions with color-coded labels (Green for correct, Red for incorrect).*

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)
*Note: The script generates a heatmap for the top 15 most frequent classes.*

## How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/TU_USUARIO/traffic-sign-classification-cnn.git](https://github.com/TU_USUARIO/traffic-sign-classification-cnn.git)
