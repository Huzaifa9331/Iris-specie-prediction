**Project_Iris Species Prediction App**

Developed By Huzaifa Masood

Project Overview

The Iris Species Prediction App is a Machine Learning project designed to classify iris flowers into one of three species based on their physical measurements.

Using a trained ML model, the application predicts whether a flower belongs to:

Iris Setosa

Iris Versicolor

Iris Virginica

The model is deployed as an interactive web application, allowing users to input flower measurements and receive instant predictions.
Project Objective:

The main objective of this project is to:

Build and train a Machine Learning classification model.

Accurately classify iris flowers into three species.

Deploy the trained model as a user-friendly web application.

Provide real-time predictions based on user input.
This project was developed using the following technologies:

Python – Core programming language

Machine Learning – Model training and prediction

TensorFlow / Keras – Neural network model development

Scikit-learn – Data preprocessing and utilities

NumPy – Numerical computations

Streamlit – Web application deployment

Project Structure
File Name	Description
app.py	Streamlit web application file
iris_model.py	Model training and saving script
iris_model.h5	Trained machine learning model
README.md	Project documentation
How to Run the Project
Step 1: Install Required Libraries

Make sure you have Python installed, then install the required libraries:

pip install streamlit tensorflow scikit-learn numpy
Step 2: Run the Application

After installing dependencies, run the Streamlit app:

streamlit run app.py
Step 3: Use the Application

Enter the flower measurements (sepal length, sepal width, petal length, petal width).

Click the Predict button.

The app will display the predicted iris species.

