# Image Recognition Project

## Introduction
The purpose of this work is to implement an image recognition system, combining conventional and unconventional machine learning techniques. The main approach is based on the K-Nearest Neighbors (KNN) algorithm for classifying images into 10 categories. Additionally, the impact of data preprocessing is evaluated through the use of an autoencoder and a pre-trained neural network model, CLIP embedding from OpenAI, on the classifier's performance.

## Description of the Dataset
The project utilizes the CIFAR-100 dataset, which includes 60,000 images categorized into 20 and 100 parent classes and subclasses, respectively. For this work, 10 subclasses were randomly selected from 9 parent categories, with each subclass containing 600 images (500 for training and 100 for evaluation). The dataset ensures equal representation across classes, with each class containing the same number of images.

- **Image Dimensions**: 32x32 pixels
- **Color Channels**: 3 (Red, Green, Blue)
- **Data Storage**: Pixels are stored in a single row with 3072 columns.

## Data Processing Before Feeding the KNN Classifier
The main approach is implemented using the K-Nearest Neighbors (KNN) algorithm, which classifies images based on their similarity to K neighboring images in the training set. The following preprocessing techniques were implemented:

1. **Image Representation with Autoencoder**:
   - An autoencoder reduces image dimensionality to 128 dimensions.
   - KNN is applied to these lower-dimensional representations.

2. **Image Representation with CLIP Embedding**:
   - A pre-trained neural network model from OpenAI extracts features from each image.
   - KNN is applied to the extracted features.

## Autoencoder Architecture
The autoencoder employs a symmetrical sequential structure of neural networks:
- **Compression Phase**: 4 layers with 1024, 512, 256, and 128 neurons.
- **Decompression Phase**: 4 layers with 256, 512, 1024, and 3072 neurons.
- **Activation Functions**: 
  - ReLU for intermediate layers
  - Sigmoid for the final layer

## CLIP Model
CLIP (Contrastive Language-Image Pretraining) is a pre-trained model designed to associate images with their corresponding text descriptions. Two approaches were employed:
1. **Few-shot approach**: Understands new image categories based on a few samples.
2. **Zero-shot approach**: Combines image representations with text descriptions to categorize images.

## Results
### Key Metrics
- **Accuracy**: Measures the percentage of correctly classified images.
- **Precision**: Measures how many of the classified instances actually belong to the class.
- **Recall**: Measures how many of the actual instances were correctly classified.

### Performance Overview
1. **Original Images without Any Processing**:
   - **Accuracy**: 46%
   - Precision and Recall for each class are provided.

2. **Representations Using Autoencoder**:
   - **Accuracy**: 43%
   - Precision and Recall for each class are provided.

3. **Categorization with the Zero-Shot Approach**:
   - **Accuracy**: 89%
   - Performance metrics for the classes are included.


## Conclusions
The results of this image recognition project demonstrate that the CLIP model significantly outperforms the K-Nearest Neighbors (KNN) algorithm when applied to image categorization tasks. Specifically, the accuracy of the CLIP model approached 90% in both few-shot and zero-shot scenarios, while the KNN algorithm achieved a maximum accuracy of 46% with unprocessed images.

These findings highlight the effectiveness of combining traditional machine learning techniques with advanced neural network architectures. However, the project also faced challenges, particularly with class similarity affecting classification accuracy among similar classes like leopard and tiger.

Future work could explore enhancing the autoencoder's performance and investigating additional data augmentation techniques to improve classification accuracy further. Additionally, integrating other deep learning models may provide further insights into the potential of image recognition systems.



## Environment Setup
To set up the project environment, you can create a conda environment with the following commands:

```bash
conda create -n image_recognition python=3.12
conda activate image_recognition
conda install pandas numpy matplotlib seaborn scikit-learn tensorflow
pip install torch torchvision transformers
```

Alternatively, you can use the following environment.yml file to recreate the environment