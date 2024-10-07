# Driver Behavior Detection

## Overview

This project implements a driver behavior detection system using deep learning models. The system classifies driver activities based on images captured while driving, utilizing architectures such as AlexNet, VGG, and ResNet. By analyzing visual data, the model aims to provide accurate classifications of driver behaviors, The models are trained to classify different activities based on images, including:

- Other Activities
- Safe Driving
- Talking on Phone
- Texting on Phone
- Turning


## Table of Contents

- [Dataset](#dataset)
- [Models](#models)
- [Dependencies](#dependencies)
- [Usage](#usage)
    [Training](#training)
    [Streamlit Deployment](#streamlit-deployment)



## Dataset
The dataset used for this project The dataset used for this project is sourced from [https://www.kaggle.com/datasets/robinreni/revitsone-5class], consists of labeled images representing various driver behaviors. It includes a diverse set of driving scenarios, ensuring robust model training and evaluation. Specific details about the dataset, including the source and preprocessing steps, can be found in the project files.

## Models
This project utilizes three different deep learning architectures for driver behavior detection:

1. **AlexNet**:
   - **Architecture**: AlexNet is a convolutional neural network (CNN) known for its depth and complexity, which enables it to learn rich features from images.
   - **Key Features**: It consists of five convolutional layers followed by fully connected layers, utilizing ReLU activation functions and dropout for regularization.
   - **Use Case**: Effective for image classification tasks, AlexNet serves as a baseline model for this project.

2. **VGG**:
   - **Architecture**: VGG is a deeper CNN architecture characterized by its use of small 3x3 convolutional filters. It has a uniform architecture that is easy to implement and modify.
   - **Key Features**: VGG includes multiple convolutional layers followed by max-pooling layers and fully connected layers. Its design promotes a hierarchical feature extraction process.
   - **Use Case**: VGG is known for achieving high performance on image classification benchmarks, making it a strong candidate for detecting driver behaviors.

3. **ResNet**:
   - **Architecture**: ResNet introduces the concept of residual learning through skip connections, allowing gradients to flow more easily during backpropagation. This enables the training of much deeper networks.
   - **Key Features**: ResNet consists of blocks with skip connections that help mitigate the vanishing gradient problem. It allows for the construction of very deep networks while maintaining performance.
   - **Use Case**: ResNet's ability to learn complex patterns in data makes it particularly effective for challenging image classification tasks, such as recognizing diverse driver behaviors.

## Dependencies
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Streamlit

## Usage
1. **Data Preprocessing**:
   - Load the dataset and perform data cleaning (deleting defect images, normalization, etc.).
   - Resize images and rescaling
   -  split the dataset into training and validation sets, where 75% of dataset for Train, 20% for Test, and 5% is for validation.

2. **Model Training**:
   - Implement and train deep learning models (AlexNet, VGG, ResNet) on the preprocessed dataset.
   - Save the trained models for later use.

3. **Streamlit App**:
   - Run the Streamlit app to interact with the models:
   But first you need to activate virtual conda environment to run the app.

   ```bash
   # Create a virtual environment
      python -m venv env

   # Activate the virtual environment
    env\Scripts\activate
   # Install required packages
    pip install tensorflow opencv-python streamlit numpy
   #Run the streamlit app
    streamlit run app.py

Select a Model: Open the web interface in your browser, select a model, and upload an image to classify the driver behavior.

4. **Model on kaggle**:
   - This model is done on kaggle [https://www.kaggle.com/code/mahmoudabdelnasser/driver-behavior-detection] take a look.

