# CICIDS2017 Cyberattack Classification Project

## Overview

This project focuses on classifying network traffic using the CICIDS2017 dataset, which includes both benign activity and various types of cyberattacks. The dataset is highly imbalanced, with benign traffic accounting for 80% and the remaining 20% representing cyberattacks such as brute force, DoS, and DDoS attacks. The goal of the project is to:

1. Classify network traffic as either benign or malicious.
2. Identify the specific type of cyberattack once classified as malicious.

## Dataset

The CICIDS2017 dataset was chosen for its realistic simulation of network activities. It contains 15 target classes with a significant imbalance between benign and attack instances. 

## Methodology

### 1. Data Preprocessing
- **Data Cleaning**: Removed rows containing invalid values like NaN or infinity.
- **Data Rescaling**: Applied Standard Scaler for normalization to ensure features are on the same scale.
- **Multicollinearity Resolution**: Identified and removed highly correlated features using a correlation matrix.

### 2. Binary Classification
The first stage focuses on classifying each observation as either benign or malicious using machine learning models. 

### 3. Attack Type Prediction
In the second stage, we applied multinomial classification to predict the specific type of attack among the malicious traffic.

## Models

Several machine learning algorithms were applied, including:

- **Multinomial Logistic Regression**
- **Gaussian Naive Bayes**
- **k-Nearest Neighbors**

## Results

The classification and attack prediction results are detailed in the Jupyter notebook, which includes feature selection, class balancing, and model performance.


## Setting Up the Environment

To run this project, you need to set up a Conda environment with the required dependencies. You can do this in one of two ways:

### Option 1: Create Environment from `environment.yaml`

 ```bash
   conda env create -f environment.yaml
   conda activate cybersecurity

```

### Option 2: Use the conda command to install the libraries:

```bash

    conda install pandas numpy matplotlib seaborn scikit-learn tqdm
```
