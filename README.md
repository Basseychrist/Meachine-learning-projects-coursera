# Titanic Survival Prediction - Machine Learning Practice

## Project Overview
This project demonstrates a complete machine learning pipeline using the famous Titanic dataset. The goal is to predict passenger survival based on various attributes like age, sex, class, etc. It serves as a practice lab for optimizing ML pipelines using Scikit-learn, preparing for real-world machine learning tasks.

## Detailed Workflow Breakdown

The notebook follows a structured approach, step-by-step:

### 1. Environment Setup
The project begins by installing and importing necessary Python libraries. Key libraries include:
- **Pandas & Numpy**: For data manipulation and numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For building the machine learning models, preprocessing, and evaluation.

### 2. Data Loading and Preparation
- **Dataset**: The Titanic dataset is loaded using Seaborn's built-in repository.
- **Feature Selection**: Relevant features (e.g., `pclass`, `sex`, `age`, `fare`) are selected, while less useful columns (like `deck` due to missing values, or `alive` which is a duplicate of the target) are dropped.
- **Target Definition**: The `survived` column is identified as the target variable (0 = No, 1 = Yes).

### 3. Exploratory Analysis & Splitting
- **Class Balance Check**: The code checks the distribution of survivors vs. non-survivors. Since only ~38% survived, the classes are slightly imbalanced.
- **Stratified Split**: The data is split into training (80%) and testing (20%) sets. `stratify=y` is used to ensure the proportion of survivors is consistent across both sets.

### 4. Advanced Preprocessing
The notebook constructs a robust preprocessing pipeline to handle different data types automatically:
- **Numerical Features**: 
  - Missing values are filled (imputed) using the **median**.
  - Data is scaled using **StandardScaler** to normalize the range of values.
- **Categorical Features**: 
  - Missing values are filled using the **most frequent** value.
  - Categories are converted into numbers using **OneHotEncoder**.
- **ColumnTransformer**: These steps are combined into a single transformer that processes numerical and categorical data in parallel.

### 5. Model 1: Random Forest Classifier
- **Pipeline Construction**: A pipeline is created combining the preprocessor and a `RandomForestClassifier`.
- **Hyperparameter Tuning**: `GridSearchCV` is used to test various combinations of parameters (like `n_estimators`, `max_depth`) using Stratified K-Fold cross-validation.
- **Evaluation**: 
  - The best model is used to predict results on the test set.
  - Performance is measured using a **Classification Report** and a **Confusion Matrix**.
- **Feature Importance**: The notebook extracts and visualizes which features (e.g., `sex`, `age`) were most influential in the Random Forest's decision-making.

### 6. Model 2: Logistic Regression
- **Model Swapping**: The pipeline is updated to replace the Random Forest with a `LogisticRegression` model, demonstrating the flexibility of Scikit-learn pipelines.
- **Re-tuning**: A new parameter grid (focusing on `penalty`, `solver`, etc.) is defined, and `GridSearchCV` is run again to find the best Logistic Regression model.
- **Comparison**: The new model is evaluated and compared against the Random Forest results.
- **Coefficient Analysis**: Instead of "importance," the magnitude and direction of the Logistic Regression coefficients are plotted to show how each feature positively or negatively affects survival odds.

## Conclusion
This notebook provides a comprehensive template for binary classification problems, covering everything from raw data to model comparison and interpretation.
# Project-Titanic-survival-prediction
