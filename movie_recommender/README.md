# Movie Recommendation System

## Overview
This project aims to develop a movie recommendation system using machine learning techniques. It analyzes user preferences based on their ratings and movie attributes to provide personalized movie suggestions. The system leverages various clustering and prediction algorithms, including K-means, DBSCAN, OPTICS, Random Forest, and Linear Regression.

## Table of Contents
- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Clustering Algorithms](#clustering-algorithms)
  - [K-means](#k-means)
  - [DBSCAN](#dbscan)
  - [OPTICS](#optics)
- [Prediction Algorithms](#prediction-algorithms)
  - [Random Forest](#random-forest)
  - [Linear Regression](#linear-regression)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
The project utilizes various algorithms to analyze user movie ratings and suggest movies that align with their preferences. It employs clustering techniques to group users and movies based on their similarities and differences in preferences.

## Data Collection
Data for the project was obtained from IMDb, focusing on movie ratings, genres, and metadata about actors, writers, and directors. An additional code file, `imdb_scraper.ipynb`, was created to facilitate the data collection process.

## Clustering Algorithms

### K-means
The K-means algorithm is employed to group users based on their ratings for popular movie genres. A sampling method was used to randomly select the top 2 most popular genres for analysis.

### DBSCAN
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters based on density and distinguishes between core points and noise. Its results were less satisfactory, with many observations classified as noise due to the heterogeneity of user preferences.

### OPTICS
OPTICS (Ordering Points To Identify the Clustering Structure) extends DBSCAN by creating a reachability plot to visualize the clustering structure. This approach provided more flexibility and scalability for datasets with variable densities.

## Prediction Algorithms

### Random Forest
Random Forest is used to predict user ratings by aggregating decisions from multiple decision trees, each considering various movie attributes. Key advantages include robustness to outliers and the ability to model nonlinear relationships, although it suffers from overfitting without proper parameter tuning.

**Evaluation Metrics:**
- Mean Squared Error (MSE): 0.7406 (Train), 0.5632 (Test)
- Mean Absolute Error (MAE): 0.6631 (Train), 0.5741 (Test)
- Accuracy: 0.2521

### Linear Regression
Linear Regression models the relationship between user ratings and movie features by finding the best-fitting line. It offers simplicity and interpretability but is limited by its linearity assumption and sensitivity to outliers.

**Evaluation Metrics:**
- Mean Squared Error (MSE): 0.83199
- Mean Absolute Error (MAE): 0.705445
- R-squared: 0.222631

## Results
The comparative analysis of Random Forest and Linear Regression shows similar predictive abilities, with Linear Regression offering better interpretability. The project concludes that the success of a movie is more influenced by the writer, followed by the actors and director.

## Conclusion
This project demonstrates the application of various clustering and prediction algorithms in a movie recommendation context. The findings highlight the significance of collaborative filtering and the importance of feature selection in model performance.

## Installation
To run this project, ensure you have Python and the necessary packages installed. You can create a virtual environment and install dependencies as follows:

```bash
# Create a virtual environment
python -m venv venv
# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r environment.yaml

or using conda install

Create the environment from the YAML file:**
conda env create -f environment.yaml
```


### Customization
Feel free to adjust any sections or add additional details specific to your project as needed!
