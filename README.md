# Stroke Prediction Analysis

This repository contains code for analyzing stroke prediction using various machine learning models, including a deep learning model and a logistic regression model.

## Introduction

Stroke prediction is a critical task in healthcare, as it can help identify individuals who are at a higher risk of experiencing a stroke. This project focuses on analyzing a dataset related to stroke prediction and building machine learning models to predict stroke occurrence based on various features.

## Libraries

Before running the code, ensure you have the following libraries installed:

- pandas
- numpy
- scipy
- matplotlib
- sklearn
- tensorflow
- seaborn

You can install them using the following command:

```bash
pip install pandas numpy scipy matplotlib scikit-learn tensorflow seaborn
```

## Dataset

The dataset used for this analysis is stored in a CSV file named `healthcare-dataset-stroke-data.csv`. The dataset contains information about various features such as age, gender, marital status, work type, residence type, smoking status, and more.

## Code Explanation

1. **Importing Libraries**: Import necessary libraries for data analysis and modeling.

2. **Loading Dataset**: Load the dataset using pandas and display its contents.

3. **Data Preprocessing**: Convert categorical columns to numerical values and handle missing data. Shuffle the dataset for randomness.

4. **Visualization**: Visualize features using line plots for better understanding.

5. **Deep Learning Model**: Build a deep learning model using TensorFlow/Keras to predict stroke occurrence. Train and evaluate the model using accuracy and a confusion matrix.

6. **Logistic Regression Model**: Build a logistic regression model using scikit-learn. Scale features and train the model. Evaluate the model's performance using a confusion matrix.

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/stroke-prediction-analysis.git
cd stroke-prediction-analysis
```

2. Make sure you have the required libraries installed.

3. Place the `healthcare-dataset-stroke-data.csv` dataset file in the project directory.

4. Run the Jupyter Notebook or Python script to execute the code.

## Results

The code will produce visualizations of features, and it will train and evaluate both a deep learning model and a logistic regression model for stroke prediction. You'll get insights into each model's performance and how well they predict strokes based on the provided dataset.

## Acknowledgments

The dataset used in this project is obtained from [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset). Make sure to credit the dataset source if you plan to use this code for your own analysis.

Feel free to modify and expand upon this project to further explore stroke prediction analysis or other healthcare-related tasks.
