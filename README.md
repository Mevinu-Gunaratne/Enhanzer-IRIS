# Enhanzer-IRIS
Code Submission

Iris Dataset Classification Project
Overview
This project demonstrates a complete machine learning workflow using the Iris dataset. It includes data exploration, preprocessing, model training, evaluation, and model saving/loading. The implementation uses a Random Forest classifier to predict the species of Iris flowers based on their sepal and petal measurements.
Requirements

Python 3.6+
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
joblib

You can install all required packages using:
pip install numpy pandas matplotlib seaborn scikit-learn joblib

Project structure
iris-classifier/
│
├── iris_classification.ipynb    # Main Jupyter notebook with the implementation
├── iris_random_forest_model.joblib  # Saved machine learning model
├── iris_feature_histograms.png  # Visualization of feature distributions
├── iris_pairplot.png           # Visualization of feature relationships
├── confusion_matrix.png        # Visualization of model performance
├── feature_importance.png      # Visualization of feature importance
└── README.md                   # This file


Implementation Details
1. Data Loading and Exploration
The project starts by loading the Iris dataset from scikit-learn and performing exploratory data analysis:

Displaying the first few rows of the dataset
Checking dimensions and data types
Confirming there are no missing values
Generating descriptive statistics
Analyzing the distribution of target classes

2. Data Visualization
Several visualizations are created to better understand the data:

Histograms showing the distribution of each feature by species
A pairplot displaying relationships between all features, colored by species

3. Data Preprocessing
The data is prepared for model training:

Features (X) and target variable (y) are separated
Data is split into training (80%) and testing (20%) sets
Stratification is used to maintain class proportions

4. Model Training
A Random Forest classifier is trained on the training data:

100 decision trees are used in the ensemble
Random state is set for reproducibility

5. Model Evaluation
The model's performance is evaluated using:

Accuracy score
Classification report (precision, recall, F1-score)
Confusion matrix
Feature importance analysis

6. Model Saving and Loading
The trained model is saved for future use:

joblib is used to save the model to disk
A demonstration of loading the model is included
The loaded model's performance is verified on test data
