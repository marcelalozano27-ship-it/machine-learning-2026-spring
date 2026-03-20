# &#x20;K Nearest Neighbor (kNN) Movie Recommendation System

**Course:** MSBA Computer Assignment 5  
**Technique:** K Nearest Neighbor (kNN)  
**Authors:** Marcela Lozano, Brandon Richard

\---

## &#x20;Project Overview

This project builds a movie recommendation system using the **K Nearest Neighbor (kNN)** algorithm. The model identifies movies that are most similar to a selected movie based on shared numeric features and returns the **Top 5 closest recommendations** using distance-based similarity.

Similarity between movies is calculated using **Euclidean distance**, where smaller distances indicate stronger similarity between films.

\---

## &#x20;Objective

The objective of this project is to:

* implement a **kNN recommendation system**
* identify the **Top 5 most similar movies** to a selected movie
* evaluate similarity using **distance metrics**
* demonstrate how feature scaling improves recommendation accuracy

\---

## &#x20;Dataset

Dataset source:

https://raw.githubusercontent.com/ArinB/MSBA-CA-Data/main/CA05/movies\_recommendation\_data.csv

The dataset contains numeric movie characteristics used to compute similarity between films.

\---

## &#x20;Methodology

The recommendation system follows these steps:

### &#x20;Import Libraries

Required Python libraries:

* NumPy
* Pandas
* Matplotlib
* Seaborn
* sklearn.neighbors
* sklearn.preprocessing

\---

### &#x20;Load Dataset

The dataset is imported directly from GitHub into the Python notebook environment.

\---

### &#x20;Select Numeric Features

Only numeric columns are used because kNN relies on distance calculations between feature values.

\---

### &#x20;Feature Scaling

The **StandardScaler** function is applied to ensure features are on comparable scales and prevent bias from variables with larger magnitudes.

\---

### &#x20;Train kNN Model

The **NearestNeighbors** function is used with:

* Euclidean distance metric
* scaled numeric features

\---

### &#x20;Generate Recommendations

A selected movie is entered as input along with its feature values.

The system:

* scales the input movie features
* calculates distances to all other movies
* returns the **Top 5 most similar movies**
* reports similarity distances

\---

## &#x20;Key Insights

* kNN is effective for recommendation systems when similarity can be defined numerically
* feature scaling is critical for accurate distance calculations
* distance metrics directly influence recommendation quality
* Euclidean distance provides interpretable similarity comparisons between movies

\---

## Limitations

* kNN can become computationally expensive with large datasets
* model performance depends heavily on feature selection
* recommendations are not personalized and remain the same for all users given the same input movie

\---

## Steps to Run the Project

Follow these steps to reproduce results:

1. Import required libraries
2. Load dataset from GitHub
3. Select numeric features
4. Apply StandardScaler
5. Train NearestNeighbors model
6. Input a movie
7. Generate Top 5 recommendations

\---

## Required Libraries

Install dependencies if needed:

pip install numpy pandas matplotlib seaborn scikit-learn

\---

## Authors

Marcela Lozano  
Brandon Richard

