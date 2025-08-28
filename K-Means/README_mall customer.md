
## Mall  Customer Segmentation


## Project Overview

This project applies unsupervised machine learning (K-Means Clustering) to segment customers of a shopping mall into distinct groups based on their demographics and spending habits. 

   These insights can help businesses tailor their marketing strategies and enhance customer experience.

   By identifying patterns within the customer base, the business can personalize marketing, optimize services, and develop targeted strategies to better meet customer needs.
## Objective
 
 -Apply clustering techniques to group customers based on shared characteristics.

-Discover hidden patterns in customer behavior without labeled data.

-Enable the mall to create customized experiences for different customer segments.



## Dataset Information

Source: Publicly available dataset (Kaggle)

Features Included:

CustomerID: Unique customer number

Gender: Male or Female

Age: Age of the customer

Annual Income (k$): Customer’s income in thousands

Spending Score (1-100): Score assigned by the mall based on customer spending and behavior
## Methodology


## Data Exploration & Cleaning

Checked for null values and outliers

Visualized feature distributions and customer profiles
## Feature Engineering & Scaling

Selected features most relevant to customer behavior

Scaled data for optimal clustering performance
## Clustering with K-Means

Used the elbow method to determine the ideal number of clusters

Built the model using KMeans from scikit-learn

## Tools used:

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn (KMeans, StandardScaler)
## Results:
Successfully segmented customers into 5 distinct clusters

Visualized customer groups by age, income, and spending patterns

Insights can be used for:

Personalized offers

Loyalty program targeting

Customer retention strategies

It gives silhouette_score 0.55 it indicate best model.
