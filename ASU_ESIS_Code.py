#!/usr/bin/env python
# coding: utf-8

# # Env Setup

# In[2]:


from google.colab import drive
drive.mount('/gdrive')
#Change current working directory to gdrive
get_ipython().run_line_magic('cd', '/gdrive')


# Install required packages

# In[3]:


get_ipython().system('pip install vecstack')


# Import required libraries

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

import warnings
warnings.filterwarnings("ignore")


# In[5]:


from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier            
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression  
from sklearn.neural_network import MLPClassifier    
from sklearn.ensemble import GradientBoostingClassifier   
from sklearn.svm import SVC


# In[6]:


from imblearn.over_sampling import SMOTE
from collections import Counter #for Smote
from vecstack import stacking


# In[7]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics


# Helper functions

# In[8]:


def read(data_file):
  return pd.read_excel(data_file)


def summary(data):
  return (pd.DataFrame({'datatype': data.dtypes, 'unique_count': data.nunique(), 'missing values':data.isna().sum(), 'missing values %':round(data.isna().sum()*100/data.shape[0],2),'unique_values': [data[x].unique() for x in data.columns]}))


def shape(data):
  print("Data has {} columns and {} rows".format(data.shape[1], data.shape[0]))

# np.split will split at 60% of the length of the shuffled array, then 80% of length (which is an additional 20% of data), 
# thus leaving a remaining 20% of the data. This is due to the definition of the function. 
def split(data):
  return np.split(data.sample(frac=1, random_state=42), [int(.6*len(data)), int(.8*len(data))])


def cal_metrics(y_org, y_pred):
  Accuracy = metrics.accuracy_score(y_org,y_pred)
  Precision = metrics.precision_score(y_org,y_pred)
  Recall = metrics.recall_score(y_org,y_pred)
  F1 = metrics.f1_score(y_org,y_pred)
  AUC = metrics.roc_auc_score(y_org,y_pred)
  print ('\nConfusion Matrix : \n', metrics.confusion_matrix(y_org, y_pred))
  print ('\nClassification Report : \n',metrics.classification_report(y_org, y_pred))
  return [Accuracy,Precision,Recall,F1,AUC]


def predict(xtrain, ytrain, x, mdl):
  mdl.fit(xtrain, ytrain)
  return mdl.predict(x)


def best_params_random(mdl, parameter,X,y):
  clf_random = RandomizedSearchCV(mdl,parameter,n_iter=25,cv=5)
  clf_random.fit(X, y)
  print(clf_random.best_params_)
  return clf_random.best_params_


def best_params_grid(mdl, parameter, X_train, y_train):
  clf_grid = GridSearchCV(mdl,parameter,cv=5)
  clf_grid.fit(X, y)
  print(clf_grid.best_params_)
  return clf_grid.best_params_


# # Data Ingestion

# Read the files and merge them 

# In[9]:


file1 = read('/gdrive/Shareddrives/ESIS/Data/ASU_2021_Year2015to2019.xlsx')
file2 = read('/gdrive/Shareddrives/ESIS/Data/ASU_2021_Year2015.xlsx')
data = pd.concat([file1, file2], ignore_index=True)
data.head()


# In[10]:


shape(data)


# In[11]:


summary(data)


# # Data Engineering

# In[58]:


clmdata = data.copy()


# Drop Duplicates

# In[59]:


clmdata.drop_duplicates(inplace=True)
shape(clmdata)


# Drop duplicates on 'Clm_Id'

# In[60]:


clmdata.drop_duplicates(subset=['Clm_Id'],inplace=True)
shape(clmdata)


# Drop NA on 'Claimant_Tenure' & 'Claimant_Age'

# In[61]:


clmdata.dropna(subset=['Claimant_Tenure','Claimant_Age'],inplace=True)


# In[62]:


clmdata['Claimant_Age'] = clmdata['Claimant_Age'].replace(r'^\s*$', 0, regex=True)
clmdata['Claimant_Age'] = clmdata['Claimant_Age'].astype(int)


# Drop all records with age less than 18

# In[63]:


clmdata = clmdata.loc[(clmdata['Claimant_Age'] >18)]


# Make textual data consistent

# In[64]:


cols = ['Clnt_ID', 'Clm_Id', 'Litigation_Ind', 'State_Code', 'Cause', 'Nature', 
         'Body_Part', 'Denial_Ind', 'Surgery_Flag', 'Hospitalization_Ind']

clmdata['Denial_Ind'] = clmdata['Denial_Ind'].fillna('n')
clmdata['Denial_Ind'] = clmdata['Denial_Ind'].replace(r'^\s*$', 'n', regex=True)

for val in cols:
  clmdata[val] = clmdata[val].apply(lambda x:x.lower())
  clmdata[val] = clmdata[val].apply(lambda x:x.strip())


# In[65]:


causes = {'object':'struck by an object','cut, puncture, or scrape': 'cut, puncture, or scrape',
          'fall':'fall or slip','collision':'collision or non-collision accidents', 'misc':'repetitive and misc causes',
          'contact':'contact related accidents','employment related losses':'employment related losses','bodily reaction':'medical related losses',
          'trans':'transportation accidents', 'commercial':'personal/commercial comprehensive','violence':'unexpected movement','aircraft':'transportation accidents', 
          'overexertion':'pressure change accidents','unknown':'unclassified wc accidents','rubbed':'unexpected movement'}

for k,v in causes.items():
  clmdata.loc[clmdata['Cause'].str.contains(k), 'Cause'] = v
clmdata['Cause']=clmdata['Cause'].str.strip()


# In[66]:


bodyParts = {'thumb':'hand','knee':'leg','hip':'back','multiple':'multiple body parts','condition':'no body part','hand':'hand','unknown':'no body part','leg':'leg',
             'wrist':'hand','ankle':'leg','upper arm':'hand','foot':'leg','elbow':'hand','toe':'leg','finger':'hand','unclassified':'no body part','forearm':'hand',
             'mult':'multiple body parts','oscalsis':'leg','eye':'head, scalp, skull','ear':'head, scalp, skull','nose':'head, scalp, skull','mouth':'head, scalp, skull','jaw':'head, scalp, skull'}

for k,v in bodyParts.items():
  clmdata.loc[clmdata['Body_Part'].str.contains(k), 'Body_Part'] = v
clmdata['Body_Part']=clmdata['Body_Part'].str.strip()


# In[67]:


clmdata['Event_Zip'] = clmdata['Event_Zip'].astype('str')
#claim_data['Event_Zip']= claim_data['Event_Zip'].fillna('000')
clmdata['Event_Zip'] = clmdata['Event_Zip'].replace(r'^\s*$', '000', regex=True)
clmdata['Event_Zip']=clmdata['Event_Zip'].str[:3]
clmdata = clmdata.loc[(clmdata['Event_Zip'] != 'nan')]


# Map yes/no to 0/1

# In[68]:


binaryCols =['Denial_Ind','Litigation_Ind','Hospitalization_Ind','Surgery_Flag']
for itm in binaryCols:
  clmdata[itm] = clmdata[itm].apply(lambda x: 1 if x == 'y' else 0)


# Populate accident month

# In[69]:


clmdata['Accident_month'] = pd.DatetimeIndex(clmdata['Accident_Date']).month


# Populate Attrny_Ind, Dispute_Ind based on given data

# In[70]:


clmdata['Dispute_Ind']=clmdata['Dispute_Recd_Dt'].apply(lambda x: 1 if not pd.isnull(x) else 0)
clmdata['Plntff_Attrny_Name']=clmdata['Plntff_Attrny_Name'].apply(lambda x: 1 if not pd.isnull(x) else 0)
clmdata['Plntff_Firm_Name']=clmdata['Plntff_Firm_Name'].apply(lambda x: 1 if not pd.isnull(x) else 0)  
clmdata['Attrny_Ind']=np.bitwise_or(clmdata['Plntff_Attrny_Name'], clmdata['Plntff_Firm_Name'])


# Drop unnecessary columns for modeling

# In[71]:


eda_data = clmdata.copy()
#eda_data.to_excel(r'/gdrive/Shareddrives/ESIS/Data/EDAdata.xlsx')
model_data = clmdata.drop(columns=['Accident_Date','Report_Date','Event_Location','Dispute_Type_Cd','Plntff_Attrny_Name','Plntff_Firm_Name','Dispute_Recd_Dt','Close_date','Clnt_ID','Clm_Id','Settlement_amt','Total_Paid','Legal_Paid_Expense'])


# In[72]:


print(eda_data.columns.to_list())
print()
print(model_data.columns.to_list())


# In[73]:


shape(model_data)


# In[74]:


print('Data loss = ',str(round((data.shape[0]-model_data.shape[0])*100/data.shape[0],2))+'%')


# # Exploratory Data Analysis

# In[77]:


eda_data.Litigation_Ind.value_counts().plot(kind='bar',figsize=(6,8),title="Litigation Indicator")
plot.show()


# Inferance: 90% of the data is not litigated and only 10% is litigated meaning the data is imbalanced 

# In[78]:


eda_data.State_Code.value_counts().plot(kind='bar',figsize=(8,6),title="State Vs Litigation")
plot.show()


# Inferance: CA is has the most litigated cases while the NJ has the least 

# In[79]:


eda_data.groupby(["Year"])['Litigation_Ind'].sum().plot(kind='bar',figsize=(10,8),title="Litigation Indicator")
plot.show()


# Inferance: There is a decline in the number of litigated cases over the years 

# In[80]:


eda_data.groupby(["Year",'State_Code'])['Litigation_Ind'].sum().unstack(['State_Code']).plot(figsize=(10,8),title="Litigation Indicator")
plot.show()


# Inferance: Texas is consistant with the numer of litigated cases and no decline is observed hence there is a need to look up the cases in texas 

# In[81]:


eda_data.corr()


# Inferance: Drop the Dispute_Ind and Attrny_Ind as they are highly correlated with the target variable

# # Data Preprocessing

# Sanity check : No missing values

# In[31]:


model_data.drop(['Dispute_Ind','Attrny_Ind'], axis=1,inplace = True)
mdl_data = model_data.copy()


# In[32]:


summary(mdl_data)


# In[33]:


categoricalFeatures = ['State_Code','Cause','Nature','Body_Part','Year','Event_Zip','Accident_month']
ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
ohe_data = pd.DataFrame(ohe.fit_transform(mdl_data[categoricalFeatures]),columns=ohe.get_feature_names(),index=mdl_data.index)
mdl_data = pd.concat([mdl_data,ohe_data],axis=1)
mdl_data.drop(labels=categoricalFeatures,axis=1,inplace=True)
mdl_data.sample(5)


# In[34]:


train, validate, test = split(mdl_data)
X_train = train.drop(['Litigation_Ind'], axis=1)
y_train = train['Litigation_Ind']
X_validate = validate.drop(['Litigation_Ind'], axis=1)
y_validate = validate['Litigation_Ind']
X_test = test.drop(['Litigation_Ind'], axis=1)
y_test = test['Litigation_Ind']


# In[35]:


shape(train)
shape(validate)
shape(test)
print('\n Train data\n',train.Litigation_Ind.value_counts())
print('\n validate data\n',validate.Litigation_Ind.value_counts())
print('\n Test data\n',test.Litigation_Ind.value_counts())


# # Data Modeling

# In[36]:


models = {'Decision Tree': DecisionTreeClassifier(),
         'Random Forest': RandomForestClassifier(),
         'Naive Bayes': GaussianNB(),
         'Logistic Regression': LogisticRegression(solver='lbfgs',max_iter=1000),
         'Gradient Boosting': GradientBoostingClassifier(),
         'Multilayer Perceptron': MLPClassifier()
         }


# In[37]:


model_score = pd.DataFrame()
for mdl_name, mdl in models.items():
  y_pred = predict(X_train, y_train, X_validate, mdl)
  print ('---------------------------------------------------------------')
  print (mdl_name,'\n')
  print ('---------------------------------------------------------------')
  model_score[mdl_name] = cal_metrics(y_validate, y_pred)
model_score.index=['Accuracy Score','Precision Score','Recall Score','F1 Score','AUC Score']
model_score


# # Hyperparameter Tuning 

# Randomized search for best value of parameters

# In[38]:


dt_parameters = {'min_samples_leaf' : range(30,200,20),'max_depth': 
            range(5,50,2),'criterion':['gini','entropy']}
rf_parameters = {'min_samples_leaf' : range(10,200,10),'max_depth': 
            range(5,100,2),'max_features':[10,20,30],'n_estimators':[20,30,40]}
gb_parameters = {'n_estimators':range(0,100,5),'learning_rate':[0.01,.1]}
mlp_parameters = {'learning_rate': ["constant", "invscaling", "adaptive"],'hidden_layer_sizes': [(100,1), (100,2), (100,3)],'activation': ["logistic", "relu", "Tanh"]}


# In[39]:


# hpt_models_rand_search = {'Decision Tree': DecisionTreeClassifier(best_params_random(DecisionTreeClassifier(),dt_parameters,X_train,y_train)),
#               'Random Forest': RandomForestClassifier(best_params_random(RandomForestClassifier(),rf_parameters,X_train,y_train)),
#               'Gradient Boosting':GradientBoostingClassifier(best_params_random(GradientBoostingClassifier(),gb_parameters,X_train,y_train)),
#               'MultiLayer Perceptron':MLPClassifier(best_params_random(MLPClassifier(),mlp_parameters,X_train,y_train))
#                }


# {'min_samples_leaf': 90, 'max_depth': 19, 'criterion': 'entropy'}
# 
# {'n_estimators': 30, 'min_samples_leaf': 10, 'max_features': 30, 'max_depth': 33}
# 
# {'n_estimators': 95, 'learning_rate': 0.1}
# 
# {'learning_rate': 'invscaling', 'hidden_layer_sizes': (100, 2), 'activation': 'relu'}
# 

# In[40]:


# hpt_models_grid_search = {'Decision Tree': DecisionTreeClassifier(best_params_grid(DecisionTreeClassifier(),dt_parameters,X_train,y_train)),
#               'Random Forest': RandomForestClassifier(best_params_grid(RandomForestClassifier(),rf_parameters,X_train,y_train)),
#               'Gradient Boosting':GradientBoostingClassifier(best_params_grid(GradientBoostingClassifier(),gb_parameters,X_train,y_train)),
#               'MultiLayer Perceptron':MLPClassifier(best_params_grid(MLPClassifier(),mlp_parameters,X_train,y_train))
#                }


# In[41]:


hpt_models = {'Decision Tree': DecisionTreeClassifier(criterion= 'entropy', max_depth= 19, min_samples_leaf= 90),
            'Random Forest': RandomForestClassifier(max_depth= 33, max_features= 30, min_samples_leaf= 10 ,n_estimators= 30),
            'Gradient Boosting':GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 95),
            'MultiLayer Perceptron':MLPClassifier(learning_rate= 'invscaling', hidden_layer_sizes=(100, 3), activation= 'relu')
            }

hpt_model_score = pd.DataFrame()
for mdl_name, mdl in hpt_models.items():
  mdl.fit(X_train, y_train)
  y_pred = mdl.predict(X_validate)
  print ('---------------------------------------------------------------')
  print (mdl_name,'\n')
  print ('---------------------------------------------------------------')
  
  Accuracy = mdl.score(X_validate,y_validate)
  Precision = metrics.precision_score(y_validate,y_pred)
  Recall = metrics.recall_score(y_validate,y_pred)
  F1 = metrics.f1_score(y_validate,y_pred)
  AUC = metrics.roc_auc_score(y_validate,y_pred)
  print ('\nConfusion Matrix : \n', metrics.confusion_matrix(y_validate, y_pred))
  print ('\nClassification Report : \n',metrics.classification_report(y_validate, y_pred))
  #crossval_score = cross_val_score(mdl, X_train, y_train, cv=10, scoring="roc_auc")
  hpt_model_score[mdl_name] = [Accuracy,Precision,Recall,F1,AUC]
hpt_model_score.index=['Accuracy Score','Precision Score','Recall Score','F1 Score','AUC Score']
hpt_model_score


# # Smote

# In[42]:


#SMOTE==============================================================================
print("___________________________________________________________________\nSMOTE\n")
print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(sampling_strategy='float', ratio=0.5)
X_res, y_res = sm.fit_resample(X_train, y_train)
print('Resampled dataset shape %s' % Counter(y_res))


# In[43]:


model_score = pd.DataFrame()
for mdl_name, mdl in models.items():
  y_pred = predict(X_res, y_res, X_validate, mdl)
  print ('---------------------------------------------------------------')
  print (mdl_name,'\n')
  print ('---------------------------------------------------------------')
  model_score[mdl_name] = cal_metrics(y_validate, y_pred)
model_score.index=['Accuracy Score','Precision Score','Recall Score','F1 Score','AUC Score']
model_score


# {'min_samples_leaf': 30, 'max_depth': 19, 'criterion': 'entropy'}
# 
# {'n_estimators': 40, 'min_samples_leaf': 10, 'max_features': 20, 'max_depth': 71}
# 
# {'n_estimators': 95, 'learning_rate': 0.1}

# In[44]:


hpt_models = {'Decision Tree': DecisionTreeClassifier(criterion= 'entropy', max_depth= 19, min_samples_leaf= 30),
            'Random Forest': RandomForestClassifier(max_depth= 71, max_features= 20, min_samples_leaf= 10 ,n_estimators= 40),
            'Gradient Boosting':GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 95),
            }

# hpt_models = {'Decision Tree': DecisionTreeClassifier(best_params_random(DecisionTreeClassifier(),dt_parameters,X_res,y_res)),
#               'Random Forest': RandomForestClassifier(best_params_random(RandomForestClassifier(),rf_parameters,X_res,y_res)),
#               'Gradient Boosting':GradientBoostingClassifier(best_params_random(GradientBoostingClassifier(),gb_parameters,X_res,y_res)),
#               'MultiLayer Perceptron':MLPClassifier(best_params_random(MLPClassifier(),mlp_parameters,X_res,y_res))
#                }

hpt_model_score = pd.DataFrame()
for mdl_name, mdl in hpt_models.items():
  mdl.fit(X_res, y_res)
  y_pred = mdl.predict(X_validate)
  print ('---------------------------------------------------------------')
  print (mdl_name,'\n')
  print ('---------------------------------------------------------------')
  
  Accuracy = mdl.score(X_validate,y_validate)
  Precision = metrics.precision_score(y_validate,y_pred)
  Recall = metrics.recall_score(y_validate,y_pred)
  F1 = metrics.f1_score(y_validate,y_pred)
  AUC = metrics.roc_auc_score(y_validate,y_pred)
  print ('\nConfusion Matrix : \n', metrics.confusion_matrix(y_validate, y_pred))
  print ('\nClassification Report : \n',metrics.classification_report(y_validate, y_pred))
  #crossval_score = cross_val_score(mdl, X_train, y_train, cv=10, scoring="roc_auc")
  hpt_model_score[mdl_name] = [Accuracy,Precision,Recall,F1,AUC]
hpt_model_score.index=['Accuracy Score','Precision Score','Recall Score','F1 Score','AUC Score']
hpt_model_score


# # Stacking

# Stacking 2 models

# In[45]:


#STACKING MODELS =====================================================================
print("___________________________________________________________________________________________\nEnsemble Methods Predictions using GradientBoosting and Decision Tree Classifier\n")

models = [ GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 95), DecisionTreeClassifier(criterion= 'entropy', max_depth= 19, min_samples_leaf= 30)]
      
S_Train, S_Test = stacking(models,                   
                           X_res, y_res, X_validate,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=metrics.f1_score, 
    
                           n_folds=2, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)


# In[46]:


#STACKING - CONTRUCT A GRADIENT BOOSTING MODEL==============================
model = GradientBoostingClassifier()
    
model = model.fit(S_Train, y_res)
y_pred = model.predict(S_Test)
print('Final prediction score for ensemble methods: [%.8f]' % metrics.accuracy_score(y_validate, y_pred))
print("Confusion Matrix after STACKING for Boosting:")
print(confusion_matrix(y_validate,y_pred))
print("=== Classification Report ===")
print(classification_report(y_validate,y_pred))


# The results of stacking 4 models and 2 models did not vary much hence we consider 2 model stacking to evaluate our resuts on test data
# 
# 

# In[47]:


#STACKING MODELS =====================================================================
print("___________________________________________________________________________________________\nEnsemble Methods Predictions using GradientBoosting and Decision Tree Classifier\n")

models = [ GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 95), DecisionTreeClassifier(criterion= 'entropy', max_depth= 19, min_samples_leaf= 30)]
      
S_Train, S_Test = stacking(models,                   
                           X_res, y_res, X_test,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=metrics.f1_score, 
    
                           n_folds=2, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)


# In[48]:


#STACKING - CONTRUCT A GRADIENT BOOSTING MODEL==============================
model = GradientBoostingClassifier()
    
model = model.fit(S_Train, y_res)
y_pred = model.predict(S_Test)
print('Final prediction score for ensemble methods: [%.8f]' % metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix after STACKING for Boosting:")
print(confusion_matrix(y_test,y_pred))
print("=== Classification Report ===")
print(classification_report(y_test,y_pred))


# The stacked model on test data is performing pretty well we observe an accuracy of 92% and weighted average f1-score of 91% 
