TASK-1: Housing Price Prediction using Linear Regression

Objective:
  The objective of this project is to predict house prices based on various features such as area, number of bedrooms, bathrooms, stories, parking, and amenities (e.g., air conditioning, main road access). The goal is to build a Linear Regression model that can estimate the price using the given dataset.

Steps Performed
1. Data Loading
  Imported the dataset Housing.csv using pandas.
  Displayed dataset info, summary statistics, and checked for missing values.

2. Data Preprocessing
  Converted categorical variables to numeric form using one-hot encoding with pd.get_dummies() (drop_first=True to avoid dummy variable trap).
  Separated features (X) and target variable (y).

3. Train-Test Split
  Split the data into 80% training and 20% testing using train_test_split.

4. Model Training
  Used Linear Regression from sklearn.linear_model to fit the training data.

5. Model Evaluation
  Predicted prices for the test set.
  Calculated Mean Squared Error (MSE) and R-squared (R²) score.
  Plotted Actual vs Predicted Prices using seaborn and matplotlib.

6. Feature Importance
  Displayed regression coefficients for each feature to understand their impact on price.

Tools & Libraries Used
  Python (Programming Language)
  pandas – Data handling and analysis
  numpy – Numerical computations
  seaborn & matplotlib – Data visualization
  scikit-learn – Machine learning model building and evaluation

Outcome
  Mean Squared Error (MSE): 1.75e+12
  R² Score: 0.6529 → The model explains ~65% of the variance in house prices.
  
Key Influencing Features:
Positive impact: airconditioning_yes, hotwaterheating_yes, bathrooms, prefarea_yes, area.
Negative impact: furnishingstatus_unfurnished, furnishingstatus_semi-furnished.
Visualization: Scatter plot showing the relationship between Actual vs Predicted Prices, with a reference line for perfect predictions.

Conclusion
This Linear Regression model provides a reasonable prediction of house prices, but performance can be improved with:
Feature engineering (e.g., interaction terms).
Using advanced models like Random Forest, XGBoost, or Gradient Boosting.
Hyperparameter tuning for better accuracy.



TASK-2: Credit Card Fraud Detection using Logistic Regression

Objective
  The goal of this project is to detect fraudulent transactions from a dataset of credit card transactions. Since the dataset is highly imbalanced, SMOTE (Synthetic Minority Over-sampling Technique) is used to handle class imbalance before training the model.

Steps Performed
1. Data Loading & Inspection
  Loaded the creditcard.csv dataset using pandas.
  Checked dataset shape, first few rows, and value counts for the Class column (0 = Not Fraud, 1 = Fraud).
  Verified there were no missing values.

2. Data Preprocessing
  Separated features (X) and target (y).
  Standardized Amount and Time columns using StandardScaler for better model performance.

3. Handling Class Imbalance
  Used SMOTE to oversample the minority class (fraud cases) and balance the dataset.

4. Train-Test Split
  Split the balanced dataset into 80% training and 20% testing sets.

5. Model Training
  Trained a Logistic Regression model (max_iter=1000) using the training data.

6. Model Evaluation
  Generated predictions for the test set.
  Calculated Precision, Recall, F1-score, and Accuracy using classification_report.
  Plotted a Confusion Matrix using seaborn for better visualization.

Tools & Libraries Used
  Python (Programming Language)
  pandas – Data handling and analysis
  numpy – Numerical computations
  matplotlib & seaborn – Data visualization
  scikit-learn – Machine learning model building and evaluation
  imbalanced-learn (imblearn) – SMOTE for handling class imbalance

Outcome
Accuracy: 95%
Precision (Fraud Class): 0.97
Recall (Fraud Class): 0.92
F1-Score (Fraud Class): 0.95
Confusion matrix shows strong classification ability for both fraud and non-fraud transactions after balancing.

Conclusion
Logistic Regression performs well after handling class imbalance with SMOTE.
The model achieves high recall for fraud detection, which is important to minimize missed fraud cases.
Future improvements could include:
Trying advanced models like Random Forest, XGBoost, or LightGBM.
Performing hyperparameter tuning for better performance.
Using feature selection to reduce dimensionality and improve speed.



TASK-3: Google Play Store Apps Data Analysis

Objective
  The aim of this project is to analyze and visualize Google Play Store application data to uncover insights about app categories, ratings, installs, pricing, reviews, and other key features. This project also demonstrates data cleaning, preprocessing, and visualization techniques using Python.

Steps Performed
1. Data Loading
  Loaded apps.csv dataset using pandas.
  Inspected the first few rows and dataset information.

2. Data Cleaning
  Removed duplicate entries.
  Dropped rows with missing values in Rating, Installs, and Category.
  Cleaned the Installs column by removing + and ,, then converted to integers.
  Converted the Price column from string format ($) to numeric format.
  Converted Reviews column to integers.
  Converted Rating column to numeric, dropping any NaNs.

3. Exploratory Data Analysis (EDA) & Visualizations
  Top 10 Categories – Bar plot of app counts per category.
  Ratings Distribution – Histogram with KDE plot to visualize rating trends.
  Size vs Rating – Scatter plot showing relationship between app size and ratings.
  Reviews vs Installs Bubble Chart – Bubbles sized by rating, log scales for better comparison.
  Word Cloud – Generated from sample review text to visualize common words.
  Interactive Box Plot – Ratings per category using Plotly for deeper exploration.

Tools & Libraries Used
  Python (Programming Language)
  pandas – Data handling and preprocessing
  numpy – Numerical computations
  matplotlib & seaborn – Data visualization
  plotly – Interactive visualizations
  wordcloud – Word cloud generation
  warnings – Ignore irrelevant warnings

Outcome
Dataset Shape (Cleaned): 8196 rows × 14 columns

Key Insights:
Certain categories dominate the Play Store with more app entries.
Most apps have ratings between 4.0 and 4.5.
Reviews and installs show a positive correlation — popular apps tend to get higher ratings.
Paid apps are fewer compared to free apps.

Conclusion
This analysis highlights the most popular categories, rating trends, and relationships between installs, reviews, and ratings.
Future improvements could include:
Sentiment analysis of actual user reviews.
Time-series analysis using Last Updated to see trends over time.
Clustering apps based on their features.
