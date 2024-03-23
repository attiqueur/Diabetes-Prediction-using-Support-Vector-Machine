# Diabetes-Prediction-using-Support-Vector-Machine

This project implements a web application for predicting whether a person is diabetic or not using a Support Vector Machine classifier. The application achieved an accuracy of 76% by detecting and removing outliers and duplicates from the dataset. Model parameters were fine-tuned using RandomizedSearchCV to optimize performance. The trained model was saved using pickle and deployed using Flask. Users can input their features through a web interface, and the prediction is displayed on the same page.

### Summary of the steps:

- Import necessary libraries 
- Loading the dataset
- Inspecting the dataset 
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Prepare data for model building
- Build the classification model
- Hyperparameter tuning 
- Save the model 
- Build a Flask web app
