# Titanic Survival Prediction

In this project, we aim to predict the survival of passengers on the Titanic using various machine learning models. We'll follow these steps:

1. **Data Preparation**: Load and explore the dataset.

2. **Data Cleaning and Preprocessing**: Handle missing values and convert categorical features into a suitable format for machine learning.

3. **Exploratory Data Analysis (EDA)**: Visualize the data to gain insights into the relationships between different features and survival.

4. **Feature Engineering**: Create new features and transform existing ones to improve model performance.

5. **Machine Learning Models**: Train and evaluate multiple machine learning models.

6. **Model Comparison**: Compare the performance of these models to select the best one for predicting passenger survival.

## Data Preparation

We'll use the following Python libraries for data processing and visualization:
- Pandas for data manipulation
- Seaborn and Matplotlib for data visualization

We load the training and testing datasets and display the first few rows to get a sense of the data.

## Data Cleaning and Preprocessing

We start by identifying and handling missing data in the dataset. Here are some key findings:
- The "Cabin" feature has a large number of missing values (77.1%).
- "Age" and "Embarked" features also have missing values.

We use various techniques to fill in missing values, such as extracting deck information from the "Cabin" feature and filling missing "Age" values with random numbers based on the mean age and standard deviation.

## Exploratory Data Analysis (EDA)

We visualize the data to understand relationships between survival and other features:
- We create histograms to compare age distributions for passengers who survived and those who did not.
- We analyze the impact of embarkation location, passenger class, and gender on survival using various plots.

## Feature Engineering

We perform feature engineering to create new variables:
- "Relatives" by combining "SibSp" (siblings/spouses) and "Parch" (parents/children) to represent the number of relatives on board.
- "Not_Alone" to identify passengers who traveled alone.
- "Age_Class" by multiplying "Age" and "Pclass" to capture the interaction between age and class.
- "Fare_Per_Person" by dividing "Fare" by the number of relatives.

## Machine Learning Models

We use a variety of machine learning models for prediction:
- Random Forest
- Logistic Regression
- K Nearest Neighbor (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Linear Support Vector Machine (SVM)

## Model Comparison

We compare the models' performance based on accuracy and select the best one. Here are the results:

- Random Forest: 92.48%
- Decision Tree: 92.48%
- KNN: 85.52%
- Support Vector Machines: 81.14%
- Logistic Regression: 81.14%
- Naive Bayes: 77.22%

The Random Forest and Decision Tree models yield the highest accuracy, both achieving 92.48%.

In this project, we successfully explored and preprocessed the Titanic dataset, created new features, and built machine learning models to predict passenger survival. The Random Forest and Decision Tree models outperformed the others in terms of accuracy.
