#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # Load Dataset

# In[2]:


df=pd.read_csv('BankChurners.csv')


# In[3]:


df


# # Basic Checks

# In[4]:


df.head()


# In[5]:


##drop unwanted column
df.drop(['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],axis=1,inplace=True)


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# # Exploratory Data Analysis

# In[9]:


# Count of churn vs. not churn
sns.countplot(data=df, x='Attrition_Flag')
plt.title("Class Distribution: Churn (1) vs No Churn (0)")
plt.show()


# #Insight:-
# In This Graph we can see that we have more data of Existing Customer and less data Attrited Customer

# # Univariate Analysis

# Numerical Feature

# In[10]:


num_cols = df.select_dtypes(include=np.number).columns.tolist()

# Histograms for numeric columns
df[num_cols].hist(figsize=(15, 12), bins=20)
plt.suptitle("Histograms of Numeric Columns")
plt.show()


# Categorical Feature

# In[11]:


cat_cols = df.select_dtypes(include='object').columns.tolist()

# Countplots for categorical features
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# # Bivariate Analysis

# Categorical vs.Target

# In[12]:


for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df,x=col,hue='Attrition_Flag')
    plt.title(f"{col} vs churn")
    plt.xticks(rotation=45)
    plt.show()


# # Data Preprocessing

# In[13]:


df.isnull().sum()


# #Insight:
# No null value Present in our data.

# # Checking Outliers:

# In[14]:


plt.figure(figsize=(25,25),facecolor='white')
plotnumber=1
for column in df:
    if plotnumber<=20:
        ax=plt.subplot(5,4,plotnumber)
        sns.boxplot(df[column])
        plt.xlabel(column,fontsize=15)
        plt.ylabel('count',fontsize=15)
        
    plotnumber+=1
plt.tight_layout    
   


# #Insight:
# We can see that Outliers are present in our data

# # Outlier Handeling:-

# In[15]:


from scipy import stats


# In[16]:


IQR = stats.iqr(df.Credit_Limit,interpolation = 'midpoint')
IQR


# In[17]:


Q1 = df.Credit_Limit.quantile(0.25)
Q3 = df.Credit_Limit.quantile(0.75)
iqr = Q3-Q1
print(iqr)


# In[18]:


min_limit=Q1-1.5*IQR
max_limit=Q3+1.5*IQR
print("min limit = ",min_limit)
print("max limit = ",max_limit)


# In[19]:


print(len(df.loc[df['Credit_Limit']>max_limit]))
df.loc[df['Credit_Limit']>max_limit]


# In[20]:


df.loc[df['Credit_Limit']<min_limit]


# In[21]:


sns.boxplot(df.Credit_Limit)


# In[22]:


df.loc[df['Credit_Limit']>max_limit,'Credit_Limit']=np.median(df.Credit_Limit)


# In[23]:


sns.boxplot(df.Credit_Limit)


# In[24]:


df.Credit_Limit=np.sqrt(df.Credit_Limit)


# In[25]:


sns.boxplot(df.Credit_Limit)


# In[26]:


from scipy import stats
columns = ['Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Ct_Chng_Q4_Q1','Months_on_book']
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


# In[27]:


plt.figure(figsize=(25,25),facecolor='white')
plotnumber=1
for column in df:
    if plotnumber<=20:
        ax=plt.subplot(5,4,plotnumber)
        sns.boxplot(df[column])
        plt.xlabel(column,fontsize=15)
        plt.ylabel('count',fontsize=15)
        
    plotnumber+=1
plt.tight_layout    


# In[28]:


df.Avg_Open_To_Buy=np.sqrt(df.Avg_Open_To_Buy)


# In[29]:


sns.boxplot(df.Avg_Open_To_Buy)


# # Handeling the categorical data

# In[30]:


df.info()


# In[31]:


df.Gender.value_counts()


# In[32]:


# use Label encoder to convert the categorical data to numerical data 


# In[33]:


from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()


# In[34]:


df.Income_Category=lc.fit_transform(df.Income_Category)


# In[35]:


df.Attrition_Flag=lc.fit_transform(df.Attrition_Flag)


# In[36]:


df


# In[37]:


df.Marital_Status.unique()


# In[38]:


# ## One hot encoding
pd.get_dummies(df['Marital_Status'],prefix='Marital_Status',dtype='int')


# In[39]:


pd.get_dummies(df['Marital_Status'],prefix='Marital_Status',dtype=int,drop_first=True)


# In[40]:


df1=pd.get_dummies(df['Marital_Status'],prefix='Marital_Status',dtype=int,drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Marital_Status'],axis=1)


# In[41]:


df


# In[42]:


df1=pd.get_dummies(df['Education_Level'],prefix='Education_Level',dtype=int,drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Education_Level'],axis=1)


# In[43]:


df1=pd.get_dummies(df['Card_Category'],prefix='Card_Category',dtype=int,drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Card_Category'],axis=1)


# In[44]:


df


# In[45]:


df['Gender']=df['Gender'].map({'M':1,'F':0})


# In[46]:


df.isnull().sum()


# In[47]:


df


# In[48]:


df.info()


# #Insight:
# No categorical data are present in our data

# In[49]:


#checking duplicate
df.duplicated().sum()


# In[50]:


# Putting feature variable to X
X = df.drop('Attrition_Flag', axis=1)

# Putting target variable to y
y = df['Attrition_Flag']


# In[51]:


y.value_counts()


# # 4. Perform Train-Test-Split

# In[52]:


# lets split the data into train and test
from sklearn.model_selection import train_test_split
# Splitting the data into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[53]:


X_train.shape,X_test.shape


# # Data balancing

# In[54]:


from imblearn.over_sampling import SMOTE

smote = SMOTE()


# In[55]:


X_smote,y_smote = smote.fit_resample(X_train,y_train)


# In[56]:


from collections import Counter
print("Actual Classes",Counter(y_train))
print("SMOTE Classes",Counter(y_smote))


# In[57]:


y_smote


# # 5.import Random forest classifier and fit the data

# In[58]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1,max_depth=5,n_estimators=100,oob_score=True)
rf_classifier.fit(X_smote,y_smote,)


# In[59]:


# Predict
from sklearn.metrics import classification_report, confusion_matrix

y_pred = rf_classifier.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[60]:


#checking the oob score
rf_classifier.oob_score_


# # This Model Gives 91% Accuracy 

# In[61]:


#Testing Accuracy 


# In[62]:


y_pred_test = y_pred
# y_pred_test[10:15]


clf_report = classification_report(y_test, y_pred_test)
print("Classification report is\n", clf_report)
print()

conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion matrix\n", conf_matrix)


# In[63]:


#Traning accuracy


# In[64]:


y_pred_train =  rf_classifier.predict(X_smote)
# y_pred_train[10:15]

clf_report = classification_report(y_smote, y_pred_train)
print("Classification report is\n", clf_report)
print()

conf_matrix = confusion_matrix(y_smote, y_pred_train)
print("Confusion matrix\n", conf_matrix)


# In[ ]:




