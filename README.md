# Chest X-Ray Diagnosis with CNN

## Overview

This project demonstrates how to build and train a convolutional neural network (CNN) model to classify chest X-ray images for disease diagnosis, particularly distinguishing between normal and pneumonia cases.

## Dataset

The dataset used in this project is a labeled collection of chest X-ray images, typically organized into three directories:

- `train`: Contains the training images.
- `test`: Contains the test images for final evaluation.
- `val`: Contains validation images used during training for model tuning.

Each image belongs to one of two categories:

- **Normal**: Healthy individuals.
- **Pneumonia**: Patients diagnosed with pneumonia.


## Project Structure

- **Model Architecture**: 
	The CNN architecture consists of several convolutional and pooling layers to capture features from X-ray images, followed by fully connected layers for classification.
    
- **Preprocessing**:
    - Images are resized to 256x256 pixels.
    - Batches of size 32 are used for training and testing.
    - Categorical labels are inferred from the directory structure.


## Training

The model is trained on the training data and evaluated on the validation data. Key steps include:

1. **Loading the Dataset**: 
	Using `keras.utils.image_dataset_from_directory` to load the images from the dataset directory.
1. **Model Construction**: 
	A sequential CNN model is built with:
    - Convolutional layers followed by ReLU activations and MaxPooling layers.
    - Dropout layers to prevent overfitting.
    - Dense layers for classification.
    
1. **Compilation**: 
	The model is compiled using the Adam optimizer and categorical cross-entropy loss function.
	
1. **Training**: 
	The model is trained using the training dataset, with validation accuracy monitored using the validation dataset.

## Evaluation

After training, the model's performance is evaluated on the test dataset to assess its ability to generalize. Metrics such as accuracy, precision, recall, and F1-score are used.

## Results

The trained model achieves an accuracy of **72%** on the test set, demonstrating its effectiveness in classifying chest X-ray images into normal and pneumonia categories.
