## Loan Prediction Project





## Project Overview
This project aims to build machine learning models that can predict whether a loan application will be **approved or not** based on applicant details.  
We experimented with multiple classification algorithms — **Logistic Regression, Support Vector Machine (SVM), and Decision Tree** — and performed **hyperparameter tuning** for each model to improve accuracy.  


## Objective
- Understand the factors that influence loan approval.  
- Build predictive models using supervised learning algorithms.  
- Compare model performance before and after hyperparameter tuning.  


## Dataset
 The dataset contains information about applicants such as:  
  - Gender, Married, Education, Self_Employed  
  - ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History  
  - Property_Area  
- Target Variable: `Loan_Status` (Y = Approved, N = Not Approved)  

## Methodology
1. Data Preprocessing 
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling (where required)  

2. Model Building 
   - Logistic Regression  
   - Support Vector Machine (SVM)  
   - Decision Tree  

3. Hyperparameter Tuning  
   - Used **GridSearchCV / RandomizedSearchCV**  
   - Tuned parameters like:  
     - Logistic Regression → `C`, `penalty`  
     - SVM → `C`, `kernel`, `gamma`  
     - Decision Tree → `max_depth`, `min_samples_split`, `criterion`  

4. Model Evaluation 
   - Metrics: Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix & Classification Report  
   - Compared base models vs tuned models  


## 📊 Results 
- Logistic Regression: 73%  
- SVM: XX% accuracy 72%  
- Decision Tree:83%

Best performing model:Decision Tree

## Tech stack
- Python 
- Pandas, NumPy, Matplotlib, Seaborn (EDA)  
- Scikit-learn (ML models + tuning)  
