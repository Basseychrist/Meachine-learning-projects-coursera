# Rainfall Prediction Classifier Project

This project builds a machine learning classifier to predict whether it will rain today in the Melbourne area based on historical weather data. The project is implemented in a Jupyter Notebook (`FinalProject_AUSWeather.ipynb`) and utilizes the Scikit-learn library for data preprocessing, modeling, and evaluation.

## Project Overview

The goal is to compare different machine learning models (Random Forest and Logistic Regression) and optimize them to predict rainfall. The workflow follows standard data science practices: data cleaning, feature engineering, pipeline construction, hyperparameter tuning, and evaluation.

## Prerequisites

The following Python libraries are required:
- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `matplotlib` & `seaborn`: For data visualization.
- `scikit-learn`: For machine learning models and preprocessing tools.

## Step-by-Step Procedure

### 1. Data Loading and Cleaning
**Procedure:**
- The dataset is loaded from a CSV file.
- Rows with missing values are dropped (`dropna()`).

**Why it is necessary:**
- Real-world data is often messy. Removing missing values is a simplification strategy to ensure the model receives complete input vectors. In more complex scenarios, imputation might be used instead.

### 2. Addressing Data Leakage
**Procedure:**
- The column `RainToday` is renamed to `RainYesterday`.
- The column `RainTomorrow` is renamed to `RainToday`.

**Why it is necessary:**
- The original dataset was designed to predict "Rain Tomorrow" using today's data. To predict "Rain Today" (e.g., deciding whether to bike to work), we must use data available *before* the event (i.e., yesterday's weather). This prevents "data leakage," where the model inadvertently sees information it shouldn't have access to during prediction.

### 3. Location Selection
**Procedure:**
- The data is filtered to include only 'Melbourne', 'MelbourneAirport', and 'Watsonia'.

**Why it is necessary:**
- Weather patterns are highly localized. A model trained on the entire continent of Australia might struggle to capture specific local microclimates. Narrowing the scope creates a more specialized and likely more accurate model for that specific region.

### 4. Feature Engineering: Seasonality
**Procedure:**
- The `Date` column is converted into a `Season` categorical feature (Summer, Autumn, Winter, Spring).
- The original `Date` column is dropped.

**Why it is necessary:**
- The exact date (e.g., "2015-03-12") is rarely predictive on its own. However, the *season* is a strong driver of weather patterns. Transforming dates into seasons simplifies the data while retaining the most predictive signal.

### 5. Data Splitting with Stratification
**Procedure:**
- The data is split into training and testing sets using `train_test_split`.
- `stratify=y` is used.

**Why it is necessary:**
- **Splitting:** We need a "holdout" test set that the model has never seen to fairly evaluate its performance.
- **Stratification:** Rainfall data is often "imbalanced" (it doesn't rain every day). Stratification ensures that the proportion of Rain/No-Rain days is the same in both the training and testing sets, preventing the model from being trained on a skewed sample.

### 6. Pipeline Construction
**Procedure:**
- **Numeric Features:** Scaled using `StandardScaler`.
- **Categorical Features:** Encoded using `OneHotEncoder`.
- **ColumnTransformer:** Applies these transformations to the correct columns automatically.

**Why it is necessary:**
- **Scaling:** Many algorithms (like Logistic Regression and SVMs) perform poorly if features have vastly different scales (e.g., Rainfall in mm vs. Pressure in hPa).
- **Encoding:** Machine learning models require numerical input. One-hot encoding converts categorical text data (like "North", "West") into binary vectors.
- **Pipeline:** Bundling these steps ensures that preprocessing is applied consistently to both training and new data, preventing errors and leakage.

### 7. Model Training: Random Forest
**Procedure:**
- A `RandomForestClassifier` is added to the pipeline.
- `GridSearchCV` is used to test combinations of hyperparameters (`n_estimators`, `max_depth`, etc.).

**Why it is necessary:**
- **Random Forest:** A robust, ensemble method that handles non-linear relationships well.
- **Grid Search:** Default model parameters are rarely optimal. Grid search systematically tests different settings to find the configuration that yields the highest accuracy.

### 8. Evaluation
**Procedure:**
- Metrics calculated: Accuracy, Confusion Matrix, Classification Report (Precision, Recall, F1-score).
- Feature Importance is extracted and plotted.

**Why it is necessary:**
- **Beyond Accuracy:** In imbalanced datasets, accuracy can be misleading (a model that always predicts "No Rain" might still have high accuracy). The Confusion Matrix and Recall (True Positive Rate) reveal how well the model actually detects rain events.
- **Feature Importance:** Helps explain *why* the model makes decisions, identifying which weather factors (e.g., Humidity, Pressure) are most critical.

### 9. Model Comparison: Logistic Regression
**Procedure:**
- The classifier in the pipeline is swapped for `LogisticRegression`.
- A new parameter grid is defined (`solver`, `penalty`, `class_weight`).
- The model is re-trained and evaluated.

**Why it is necessary:**
- Different algorithms have different strengths. Logistic Regression provides a linear baseline and is often easier to interpret. Comparing models allows us to choose the best performer for the specific task. Using `class_weight='balanced'` specifically helps address the imbalance issue identified earlier.
