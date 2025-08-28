#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import  warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('loan_approved.csv')


# In[3]:


df


# # Basic checks

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.describe()


# In[7]:


df.describe(include='O')


# #Number of missing values in each column

# In[8]:


df.info()


# #missing values are present in data

# In[9]:


df.rename(columns={"Loan_Status (Approved)":"Loan_Status"},inplace=True)


# In[10]:


df.head()


# In[11]:


df.Loan_Status.value_counts()


# # Domain Analysis:-

# These data presence across all urban,semi urban and rural area.customer first apply for loan after that company validates the customer eligibility for loan,based on customer detail provided like Gender,Marrital status,Education,Number of dependent ,Income,Credit history and others.
# 
# 
# 
# This is a Standard supervised classification task.A classification problem where we have to predict whether a loan would be approved or not
# 
# Loan_ID-Is a unique identifier assigned to each loan application in the data.
# 
# Gender-the Gender column represents the applicant’s gender, and it is typically used as a categorical feature in the dataset.
# 
# Married-the Married column indicates whether the loan applicant is married or not.
# 
# Dependent-High number of dependents might indicate more financial burden
# 
# Education- the Education column indicates the educational qualification of the loan applicant.
# 
# ApplicantIncome-represents the monthly income of the primary loan applicant.
# 
# CoapplicantIncome-represents the monthly income of the co-applicant.
# 
# LoanAmount-represents the amount of money the applicant has applied for as a loan.
# 
# Loan_Amount_Term-represents the duration of the loan in months i.e., how long the applicant will take to repay the loan.
# 
# Credit_History- indicates whether the loan applicant has a history of repaying past loans on time.
#  
# property area-represent the type of area in which the applicant's property is located.
#        i.e Urban,Ruaral,semiUrban.
# Loan_Status-indicates whether a loan application was approved or rejected.
# 

# #insights:-
# data is unbalance it baised towards loan approved yes data

# # EXPLORATORY DATA ANALYSIS

# In[12]:


df.head()


# # Univariate Analysis

# .hist plot is used to check distribution
# .count plot is used to check whether data is balanced or not

# In[13]:


sns.countplot(x=df['Loan_Status'])


# #insight:-1) clearly data is imbalanced 
# 2)since we have more recoard of loan approved or less data of No loan approved.
# 

# In[14]:


sns.countplot(x='Gender',hue='Loan_Status',data=df)


# # insight
# In these data we have to seen Loan approve is more in male as compaired to female

# In[15]:


get_ipython().system('pip install sweetviz')


# In[16]:


import sweetviz as sv #  library for univariant analysis

my_report = sv.analyze(df)## pass the original dataframe

my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"


# # Bivariate Analysis

# In[17]:


df.info()


# In[18]:


data1=df[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']]
data2=df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]


# In[19]:


plt.figure(figsize=(20,25), facecolor='white')#To set canvas 
plotnumber = 1#counter

for column in data1:#accessing the columns 
    if plotnumber<=6:
        ax = plt.subplot(3,2,plotnumber)
        sns.countplot(x=data1[column],hue=df['Loan_Status'])
        plt.xlabel(column,fontsize=20)#assign name to x-axis and set font-20
        plt.ylabel('Loan Status',fontsize=20)
    plotnumber+=1#counter increment
plt.tight_layout()


# In[20]:


plt.figure(figsize=(20,25), facecolor='white')#To set canvas 
plotnumber = 1#counter

for column in data2:#accessing the columns 
    if plotnumber<=16 :
        ax = plt.subplot(3,3,plotnumber)
        sns.histplot(x=data2[column],hue=df['Loan_Status'])
        plt.xlabel(column,fontsize=20)#assign name to x-axis and set font-20
        plt.ylabel('Loan Status',fontsize=20)
    plotnumber+=1#counter increment
plt.tight_layout()


# In[ ]:





# # Data Preprocessing and feature Engineering

# In[21]:


#finding missing value 
df.isnull().sum()


# In[22]:


df.shape


# In[23]:


## Getting the rows where values are missed in Gender features
print(len(df.loc[df['Gender'].isnull()==True]))
df.loc[df['Gender'].isnull()==True]


# In[24]:


## Checking the distribution along the both labels
df.Gender.value_counts()


# In[25]:


df.Gender.isnull().sum()/len(df)*100


# In[26]:


import seaborn as sns
sns.countplot(x='Gender',hue='Loan_Status',data=df)


# In[27]:


df.loc[df['Gender'].isnull()==True,'Gender']='Male'


# In[28]:


df.Gender.isna().sum()


# In[29]:


print(len(df.loc[df['Dependents'].isnull()==True]))
df.loc[df['Dependents'].isnull()==True]


# In[30]:


df.Dependents.value_counts()


# In[31]:


sns.countplot(x='Dependents',data=df,hue='Loan_Status')


# In[32]:


pd.crosstab(df.Dependents,df.Loan_Status)


# In[33]:


df.loc[df['Dependents'].isnull()==True,'Dependents']='3+'


# In[34]:


df.isnull().sum()


# In[35]:


df.loc[df['Married'].isnull()==True]


# In[36]:


sns.countplot(x='Married',data=df,hue='Loan_Status')


# In[37]:


df.Married.value_counts()


# In[38]:


df.loc[df['Married'].isnull()==True,'Married']='No'


# In[39]:


print(len(df.loc[df['Self_Employed'].isnull()==True]))
df.loc[df['Self_Employed'].isnull()==True]


# In[40]:


df.Self_Employed.value_counts()


# In[41]:


sns.countplot(x='Self_Employed',hue='Loan_Status',data=df)


# In[42]:


df.loc[df['Self_Employed'].isnull()==True,'Self_Employed']='No'


# In[43]:


df.isna().sum()


# In[44]:


## Histogram since it has numerical value
df.LoanAmount.hist()
plt.show()


# # Insight
# Since data is skewed, we can use median to replace the nan value. It is recommended to use mean only for symmetric data distribution.

# In[45]:


np.median(df.LoanAmount.dropna(axis=0))


# In[46]:


# Replace the nan values in LoanAmount column with median value
# data.loc[data['LoanAmount'].isnull()==True,'LoanAmount']=128.0
df.loc[df['LoanAmount'].isnull()==True,'LoanAmount']=np.median(df.LoanAmount.dropna(axis=0))


# In[47]:


df.isna().sum()


# In[48]:


df.Loan_Amount_Term.value_counts()


# In[49]:


df.Loan_Amount_Term.median()


# In[50]:


df.Loan_Amount_Term.hist()


# In[51]:


df.loc[df['Loan_Amount_Term'].isnull()==True,'Loan_Amount_Term']=np.median(df.Loan_Amount_Term.dropna(axis=0))


# In[52]:


df.Credit_History.value_counts()


# In[53]:


sns.countplot(x='Credit_History',hue='Loan_Status',data=df)


# In[54]:


df.loc[df['Credit_History'].isnull()==True,'Credit_History']=0.0


# In[55]:


df.isnull().sum()


# In[56]:


df.head()


# In[57]:


plt.figure(figsize=(25,25),facecolor='white')
plotnumber=1
for column in df:
    if plotnumber<=13:
        ax=plt.subplot(4,4,plotnumber)
        sns.boxplot(df[column])
        plt.xlabel(column,fontsize=15)
        plt.ylabel('count',fontsize=15)
        
    plotnumber+=1
plt.tight_layout    
   


# Outlier Handeling

# In[58]:


from scipy import stats


# In[59]:


IQR=stats.iqr(df.ApplicantIncome,interpolation = 'midpoint')
IQR


# In[60]:


Q1 = df.ApplicantIncome.quantile(0.25)
Q3 = df.ApplicantIncome.quantile(0.75)
iqr=Q3-Q1
print(iqr)


# In[61]:


min_limit=Q1 - 1.5*IQR
max_limit=Q3 +1.5*IQR
print("min limit = ",min_limit)
print("max limit = ",max_limit)


# In[62]:


df.loc[df['ApplicantIncome']>max_limit]


# In[63]:


df.loc[df['ApplicantIncome']<min_limit]


# In[64]:


sns.boxplot(df.ApplicantIncome)


# In[65]:


df.loc[df['ApplicantIncome'] > max_limit,'ApplicantIncome']=np.median(df.ApplicantIncome)


# In[66]:


sns.boxplot(df.ApplicantIncome)


# In[67]:


df.ApplicantIncome=np.sqrt(df.ApplicantIncome)


# In[68]:


sns.boxplot(df.ApplicantIncome)


# In[69]:


from scipy import stats
columns = ['CoapplicantIncome','LoanAmount','Loan_Amount_Term']
for col in columns:
    #caluculate IQR
    IQR=stats.iqr(df[col],interpolation = 'midpoint')
    
    #calculate Q1 and Q3
    Q1 = df[col].quantile(0.25)
    Q3 =df[col].quantile(0.75)
    
    #calculate bounds
    lb=Q1 - 1.5*IQR
    ub=Q3 +1.5*IQR
    
     
    # print summary
    print(f"\ncolumn: {col}")
    print("IQR =", IQR)
    print("Q1 =", Q1)
    print("Q3 =", Q3)
    print("lower bound = ", lb)
    print("upper bound = ", ub)
    print("outliers above upper bound:",len(df.loc[df[col] > ub]))
    print("outliers below lower bound:",len(df.loc[df[col] < lb]))
    
    #replace outliers with median
    median_val = np.median(df[col])
    df.loc[(df[col]>ub)|(df[col] <lb),col] = median_val
    print(len(df.loc[(df[col]>ub)|(df[col] <lb)]))


# In[70]:


plt.figure(figsize=(25,25),facecolor='white')
plotnumber=1
for column in df:
    if plotnumber<=13:
        ax=plt.subplot(4,4,plotnumber)
        sns.boxplot(df[column])
        plt.xlabel(column,fontsize=15)
        plt.ylabel('count',fontsize=15)
        
    plotnumber+=1
plt.tight_layout    
   


# In[71]:


df.LoanAmount=np.sqrt(df.LoanAmount)


# In[72]:


sns.boxplot(df.LoanAmount)


# In[73]:


## Step 2 Handling the categorical data
df.info()


# In[74]:


df.Gender.value_counts()


# In[75]:


df.head()


# In[76]:


## Using label encoder to convert the categorical data to numerical data
## Donot run this code.This is just implementation of label encoder.This dataset have lots relationship with target.

from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()


# In[77]:


df.Gender=lc.fit_transform(df.Gender)


# In[78]:


df


# In[79]:


df.Married.unique()


# In[80]:


df


# In[81]:


# ## One hot encoding
pd.get_dummies(df['Married'],prefix='Married',dtype='int')


# In[82]:


pd.get_dummies(df['Married'],prefix='Married',dtype=int,drop_first=True)


# In[83]:


df1=pd.get_dummies(df['Married'],prefix='Married',dtype=int,drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Married'],axis=1)


# In[84]:


df


# In[85]:


df1=pd.get_dummies(df['Education'],prefix='Education',dtype=int,drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Education'],axis=1)


# In[86]:


df1=pd.get_dummies(df['Property_Area'],prefix='Property_Area',dtype=int,drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Property_Area'],axis=1)


# In[87]:


df1=pd.get_dummies(df['Dependents'],prefix='Dependents',dtype=int,drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Dependents'],axis=1)


# In[88]:


df1=pd.get_dummies(df['Self_Employed'],prefix='Self_Employed',dtype=int,drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Self_Employed'],axis=1)


# In[89]:


df.head()


# In[90]:


df.describe()


# In[91]:


##step 5:-scaling the data
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]=\
sc.fit_transform(df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']])


# In[92]:


df.describe()


# In[93]:


## checking the duplicate rows

#data.duplicate()
df.duplicated().sum()


# In[94]:


df


# In[95]:


df.Loan_Status=df.Loan_Status.map({'Y':1,'N':0})


# In[96]:


df


# In[97]:


df


# In[98]:


## Saving the preprocessed data.
df.to_csv('MyNewPreprocessed_data.csv')


# In[99]:


pwd


# In[100]:


ls


# In[101]:


## Loading the data

preprcessed_data=pd.read_csv('MyNewPreprocessed_data.csv')


# In[102]:


preprcessed_data


# # Feature Selection

# In[103]:


preprcessed_data.head()


# # Feature Selection

# In[104]:


##create independent and dependent varibale
# Removing redundant columns
#We can drop loan id.  
l1=['Unnamed: 0','Loan_ID']
preprcessed_data.drop(l1,axis=1,inplace=True)


# In[105]:


preprcessed_data


# In[106]:


preprcessed_data.Loan_Amount_Term.value_counts()


# In[107]:


corr_data=preprcessed_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']]


# In[108]:


corr_data


# In[109]:


sns.heatmap(corr_data.corr(),annot=True)


# In[110]:


## There is no relationship among the numerical data 


# In[111]:


corr_data.describe()


# # Model Creation

# In[112]:


preprcessed_data.head(3)


# In[113]:


preprcessed_data.columns


# In[114]:


X=preprcessed_data.loc[:,['Gender', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History',  'Married_Yes',
       'Education_Not Graduate', 'Property_Area_Semiurban',
       'Property_Area_Urban', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
       'Self_Employed_Yes']]
y=preprcessed_data.Loan_Status


# In[115]:


X


# In[116]:


y


# In[117]:


##creating traning and testing data.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)


# In[118]:


## balancing the data
preprcessed_data.Loan_Status.value_counts()


# In[119]:


# !pip install imblearn


# In[120]:


from imblearn.over_sampling import SMOTE
smote= SMOTE()


# In[121]:


X_smote, y_smote = smote.fit_resample(X_train,y_train)


# In[122]:


from collections import Counter
print("Actual Classes",Counter(y_train))
print("SMOTE Classes",Counter(y_smote))


# In[123]:


#model creation
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_smote,y_smote)


# In[124]:


y_pred=clf.predict(X_test)


# In[125]:


y_pred


# In[126]:


y_test


# In[127]:


y_pred_prob=clf.predict_proba(X_test)


# In[128]:


y_pred_prob


# In[129]:


df['Loan_Status'].value_counts()


# In[130]:


#Evaluation of model


# In[131]:


from  sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,f1_score


# In[132]:


cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[133]:


recall=recall_score(y_test,y_pred)
recall


# In[134]:


precision=precision_score(y_test,y_pred)
precision


# In[135]:


f1score=f1_score(y_test,y_pred)
f1score


# In[136]:


acurracy=accuracy_score(y_test,y_pred)
acurracy


# In[137]:


cr=classification_report(y_test,y_pred)
print(cr)


# Testing Data Accuracy

# In[138]:


y_pred_test = y_pred


test_accuracy = accuracy_score(y_test, y_pred_test)
print("Testing accuracy is :", test_accuracy)
print()
clf_report = classification_report(y_test, y_pred_test)
print("Classification report is\n", clf_report)
print()

conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion matrix\n", conf_matrix)


# Traning Data Accuracy

# In[139]:


y_pred_train = clf.predict(X_smote)


train_accuracy = accuracy_score(y_smote, y_pred_train)
print("Training accuracy is :", train_accuracy)
print()
clf_report = classification_report(y_smote, y_pred_train)
print("Classification report is\n", clf_report)
print()

conf_matrix = confusion_matrix(y_smote, y_pred_train)
print("Confusion matrix\n", conf_matrix)


# # SVM 

# In[140]:


from sklearn.svm import SVC
svclassifier = SVC() ## base model with default parameters
svclassifier.fit(X_smote, y_smote)


# In[141]:


# Predict output for X_test
y_hat=svclassifier.predict(X_test)


# In[142]:


## evaluating the model created
from sklearn.metrics import accuracy_score,recall_score,precision_score,classification_report,f1_score
acc=accuracy_score(y_test,y_hat)
acc


# In[143]:


# Calssification report measures the quality of predictions. True Positives, False Positives, True negatives and False Negatives 
# are used to predict the metrics of a classification report 

print(classification_report(y_test,y_hat))


# In[144]:


cm1=pd.crosstab(y_test,y_hat)
cm1


# In[145]:


# F1 score considers both Precision and Recall for evaluating a model
f1=f1_score(y_test,y_hat)
f1


# In[146]:


## checking cross validation score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svclassifier,X,y,cv=10,scoring='f1')
print(scores)
print("Cross validation Score:",scores.mean())
print("Std :",scores.std())
#std < 0.05 is good. 


# # GridSearchCV

# In[147]:


from sklearn.model_selection import GridSearchCV
  
# Defining parameter range
param_grid = {'C': [5, 10,50,60,70], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
             'random_state':(list(range(1, 20)))}
model=SVC()
grid = GridSearchCV(model, param_grid, refit = True, verbose = 4,scoring='f1',cv=5)
  
# fitting the model for grid search
grid.fit(X,y)


# In[148]:


# print best parameter after tuning
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
#print(grid.best_estimator_)


# In[149]:


# clf=SVC(C=100, gamma=0.001,random_state=42) ##0.1
clf=SVC(C=5, gamma=0.1,random_state=1) ##0.1


# In[150]:


clf.fit(X_smote, y_smote)


# In[151]:


y_clf=clf.predict(X_test)


# In[152]:


print(classification_report(y_test,y_clf))


# In[153]:


cm=pd.crosstab(y_test,y_clf)
cm


# In[154]:


f1=f1_score(y_test,y_clf)
f1


# In[155]:


scores_after = cross_val_score(clf,X,y,cv=10,scoring='f1')
print(scores_after)
print("Cross validation Score:",scores_after.mean())
print("Std :",scores.std())
#std of < 0.05 is good. 


# # Decision Tree Classifier

# In[156]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


# In[157]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_smote, y_smote)


# Testing Data Accuracy

# In[158]:


y_pred_test = dt_model.predict(X_test)
y_pred_test[10:15]

test_accuracy = accuracy_score(y_test, y_pred_test)
print("Testing accuracy is :", test_accuracy)
print()
clf_report = classification_report(y_test, y_pred_test)
print("Classification report is\n", clf_report)
print()

conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion matrix\n", conf_matrix)


# Traning Data Accuracy

# In[159]:


y_pred_train = dt_model.predict(X_smote)
y_pred_train[10:15]

train_accuracy = accuracy_score(y_smote, y_pred_train)
print("Training accuracy is :", train_accuracy)
print()
clf_report = classification_report(y_smote, y_pred_train)
print("Classification report is\n", clf_report)
print()

conf_matrix = confusion_matrix(y_smote, y_pred_train)
print("Confusion matrix\n", conf_matrix)


# # GridSearchCV

# In[160]:


hyperparameters = {'criterion' : ["gini", "entropy"],
                   'max_depth' : np.arange(2,10),
                   'min_samples_split': np.arange(2,10),
                   "min_samples_leaf" : np.arange(2,10)
                  }
dt_model = DecisionTreeClassifier()
gscv_dt_model = GridSearchCV(dt_model, hyperparameters, cv=5)
gscv_dt_model.fit(X_smote, y_smote)


# In[161]:


gscv_dt_model.best_params_


# In[165]:


best_model = DecisionTreeClassifier(criterion= 'entropy',
                     max_depth= 8,
                     min_samples_leaf= 3,
                        min_samples_split= 4)
best_model.fit(X_smote, y_smote)


# Testing Data accuracy

# In[166]:


y_pred_test = best_model.predict(X_test)
y_pred_test[10:15]

test_accuracy = accuracy_score(y_test, y_pred_test)
print("Testing accuracy is :", test_accuracy)
print()
clf_report = classification_report(y_test, y_pred_test)
print("Classification report is\n", clf_report)
print()

conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion matrix\n", conf_matrix)


# Testing Data Accuracy

# In[167]:


y_pred_train = best_model.predict(X_smote)
y_pred_train[10:15]

train_accuracy = accuracy_score(y_smote, y_pred_train)
print("Training accuracy is :", train_accuracy)
print()
clf_report = classification_report(y_smote, y_pred_train)
print("Classification report is\n", clf_report)
print()
conf_matrix = confusion_matrix(y_smote, y_pred_train)
print("Confusion matrix\n", conf_matrix)


# In[ ]:




