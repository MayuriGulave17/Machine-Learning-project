
# Credit Card Churn Prediction



##  Project Overview 
Customer churn is a critical challenge in the banking and financial services industry. This project aims to predict whether a credit card customer is likely to leave the bank (churn) using historical customer data. By identifying customers at risk of attrition, businesses can proactively implement retention strategies.

In This project  We uses the BankChurners.csv dataset from Kaggle and applies machine learning techniques—primarily Random Forest Classifier—to build a predictive model.



## Objective
-Analyze and understand the factors that contribute to customer churn.

-Predict whether a credit card customer will churn (leave the bank) or stay.

-Evaluate and improve model performance using various  metrics.

##  Project Overview 
Customer churn is a critical challenge in the banking and financial services industry. This project aims to predict whether a credit card customer is likely to leave the bank (churn) using historical customer data. By identifying customers at risk of attrition, businesses can proactively implement retention strategies.

In This project  We uses the BankChurners.csv dataset from Kaggle and applies machine learning techniques—primarily Random Forest Classifier—to build a predictive model.



## Data Describtion
Source: Kaggle – Credit Card Customers Dataset

The Dataset Includes:

-Customer demographics (age, gender, dependents)

-Credit usage behavior (limit, revolving balance, utilization ratio)

-Transaction activity (amounts, frequency)

-Engagement indicators (months on book, number of contacts)

Target Variable:

Attrition_Flag:

Attrited Customer → churned

Existing Customer → active customer


## Exploratory Data Analysis(EDA)
-Target variable distribution (class imbalance)

-Univariate and bivariate feature analysis
## Data prepossesing
Removed unnecessary columns (e.g., customer ID, Naive Bayes outputs)

Handeling Outliers

Handled categorical features using label encoding

Handle class imbalance using SMOTE 

## Feature Selection
Very  firstly We Select features variable in X and target variable in Y
 
 ## Perform Train Test Split
  We Split A Data into Train  and Test
  
# Model Building
We build Random Forest Classifier Model 

Evaluated model using confusion matrix, precision, recall, and F1-score

Explored feature importance

#Conclusion:


## Feature Selection
Very  firstly We Select features variable in X and target variable in Y
 
 ## Perform Train Test Split
  We Split A Data into Train  and Test
  
# Model Building
We build Random Forest Classifier Model 

Evaluated model using confusion matrix, precision, recall, and F1-score

Explored feature importance

#Conclusion:


## Conclusion:

Achieved strong classification performance with Random Forest

Identified key features contributing to customer churn

Feature importance revealed Total_Trans_Ct, Total_Trans_Amt, and Credit_Limit as top predictors

       "The model achieves 91% accuracy."
