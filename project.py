#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 23:33:13 2019

@author: Pappa
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn import model_selection
from sklearn import linear_model
from sklearn import feature_selection
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV


def printresult(actual,predicted):
    confmatrix=metrics.confusion_matrix(actual,predicted)
    accscore=metrics.accuracy_score(actual,predicted)
    precscore=metrics.precision_score(actual,predicted)
    recscore=metrics.recall_score(actual,predicted)
    print(confmatrix)
    print("accuracy : {:.4f}".format(accscore))
    print("precision : {:.4f}".format(precscore))
    print("recall : {:.4f}".format(recscore))
    print("f1-score : {:.4f}".format(metrics.f1_score(actual,predicted)))
    print("AUC : {:.4f}".format(metrics.roc_auc_score(actual,predicted)))
    
    
    
    
df=pd.read_csv("/home/delta/dataset/project/credit_default.csv")
df.info()

df.head()

plt.figure(figsize=(8,8))
sns.heatmap(cbar=False,annot=True,data=df.corr()*100)
plt.title('% Corelation Matrix')
plt.show()


colname={"ID":"id","EDUCATION":"edu","MARRIAGE":"mar","LIMIT_BAL":"lbal","PAY_0":"p1","PAY_2":"p2","PAY_3":"p3","PAY_4":"p4","PAY_5":"p5","PAY_6":"p6",
         "BILL_AMT1":"ba1","BILL_AMT2":"ba2","BILL_AMT3":"ba3","BILL_AMT4":"ba4","BILL_AMT5":"ba5","BILL_AMT6":"ba6",
         "PAY_AMT1":"pa1","PAY_AMT2":"pa2","PAY_AMT3":"pa3","PAY_AMT4":"pa4","PAY_AMT5":"pa5","PAY_AMT6":"pa6","default.payment.next.month":"dpay" }

df.rename(columns=colname,inplace=True)
df.info()

df[["p1","p2","p3","p4","p5","p6"]].describe()

df[["p1","p2","p3","p4","p5","p6"]].describe()


df.drop(df["bp"]==0,axis=0,inplace=True)    
    
for index,record in df.iterrows():
    if record["mar"]==0:
        df.    
      
for index,record in df.iterrows():
    if record["bmi"]==0:
        df.drop(index,axis=0,inplace=True)       
    
df.info() #727 non-null int64  
df.describe()




################



df.mar.replace({0:3},inplace=True)


change = (df.edu == 5) | (df.edu == 6) | (df.edu == 0)
df.loc[change, 'edu'] = 4
#df.edu.value_counts()


change = (df.p1 == -1) | (df.p1 == -2) | (df.p1 == 0)
df.loc[change, 'p1'] = 0
#df.p2.value_counts()

change = (df.p2 == -1) | (df.p2 == -2) | (df.p2 == 0)
df.loc[change, 'p2'] = 0
#df.p2.value_counts()

change = (df.p3 == -1) | (df.p3 == -2) | (df.p3 == 0)
df.loc[change, 'p3'] = 0
#df.p3.value_counts()

change = (df.p4 == -1) | (df.p4 == -2) | (df.p4 == 0)
df.loc[change, 'p4'] = 0
#df.p4.value_counts()

change = (df.p5 == -1) | (df.p5 == -2) | (df.p5 == 0)
df.loc[change, 'p5'] = 0from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
#df.p5.value_counts()

change = (df.p6 == -1) | (df.p6 == -2) | (df.p6 == 0)
df.loc[change, 'p6'] = 0
#df.p6.value_counts()


ddf=df[["SEX","edu","mar","AGE","p1","p2","p3","p4","p5","p6","dpay"]]
ddf.info()
ddf['agecat'] = 0 #creates a column of 0
ddf.loc[((ddf['AGE'] >= 20) & (ddf['AGE'] < 30)) , 'agecat'] = 1
ddf.loc[((ddf['AGE'] >= 30) & (ddf['AGE'] < 40)) , 'agecat'] = 2
ddf.loc[((ddf['AGE'] >= 40) & (ddf['AGE'] < 50)) , 'agecat'] = 3
ddf.loc[((ddf['AGE'] >= 50) & (ddf['AGE'] < 60)) , 'agecat'] = 4
ddf.loc[((ddf['AGE'] >= 60) & (ddf['AGE'] < 70)) , 'agecat'] = 5
ddf.loc[((ddf['AGE'] >= 70) & (ddf['AGE'] < 81)) , 'agecat'] = 6



df.info()



ddf.info()


ddf1=ddf.copy()
ddf1.info()

######
for col in ["SEX","edu","mar","p1","p2","p3","p4","p5","p6","agecat"]:
    for val in ddf[col].unique():
        ddf1[col][(ddf1[col] == val) & (ddf.dpay==0)] = (ddf[(ddf[col]==val) & (ddf.dpay==0)].shape[0])/ddf[ddf[col]==val].shape[0]
        ddf1[col][(ddf1[col] == val) & (ddf.dpay==1)] = (ddf[(ddf[col]==val) & (ddf.dpay==1)].shape[0])/ddf[ddf[col]==val].shape[0]




ddf1.sample(10)

X=ddf1[["SEX","p1","p2","p3","agecat"]]
y=ddf1["dpay"]
#####grid search
X.info()
y.info()
depth=list(range(3,11))
grid = {'max_depth': np.arange(3, 10),
             'criterion' : ['gini','entropy'],
             'max_leaf_nodes': [5,10,20,100],
             'min_samples_split': [2, 5, 10, 20]}

model=tree.DecisionTreeClassifier()
gmodel=model_selection.GridSearchCV(model,grid,scoring="recall")
gmodel.fit(X,y)
best = gmodel.best_estimator_

best.fit(X,y)
print(dict(zip(X.columns, best.feature_importances_)))



X=ddf1[["SEX","edu","mar","agecat","p1","p2","p3","p4","p5","p6"]]
Y=ddf1["dpay"]

model =linear_model.LogisticRegression()
rfe = feature_selection.RFE(best,15)
fit = rfe.fit(X, y)
print("Num Features: %d",fit.n_features_)
print("Selected Features: %s",fit.support_)
print("Feature Ranking: %s",fit.ranking_)


df.info()
cdf=df[["lbal","ba1","ba2","pa1","ba3","pa2","ba4","pa3","ba5","pa4","ba6","pa5","pa6"]]
cdf.describe()

X=cdf.copy()
######robust scaler
model=preprocessing.RobustScaler()
model.fit(X)
y=model.transform(X)
y.reshape()


rodf = pd.DataFrame(y, columns=["lbal","ba1","ba2","pa1","ba3","pa2","ba4","pa3","ba5","pa4","ba6","pa5","pa6"])
rodf.sample(10)
rodf.info()
X=rodf.copy()
X["due2"]=rodf.ba3-rodf.pa2
X.info()
y=df["dpay"]
depth=list(range(3,11))
grid = {'max_depth': np.arange(3, 20),
             'criterion' : ['gini','entropy'],
             'max_leaf_nodes': [5,10,20,100],
             'min_samples_split': [2, 5, 10, 20]}

model=tree.DecisionTreeClassifier()
gmodel=model_selection.GridSearchCV(model,grid,scoring="recall")
gmodel.fit(X,y)
best = gmodel.best_estimator_

best.fit(X,y)
print(dict(zip(X.columns, best.feature_importances_)))

#####applying model decision tree and random froest ,byes

ddf1.sample(10)
rodf.sample(2)

X=1
X=ddf1[["SEX","edu","mar","agecat","p1","p2","p3","p4","p5","p6"]]
X[["rlbal","rba1","rba2","rpa1","rba3","rpa2","rba4","rpa3","rba5","rpa4","rba6","rpa5","rpa6"]]=rodf[["lbal","ba1","ba2","pa1","ba3","pa2","ba4","pa3","ba5","pa4","ba6","pa5","pa6"]]
X.info()
y=df["dpay"]
print(best)
# =============================================================================
# {'SEX': 0.0, 'edu': 0.0, 'mar': 1.0, 'agecat': 0.0, 'p1': 0.0, 'p2': 0.0, 'p3': 0.0, 'p4': 0.0, 'p5': 0.0, 'p6': 0.0, 'rlbal': 0.0, 'rba1': 0.0, 'rba2': 0.0, 'rpa1': 0.0, 'rba3': 0.0, 'rpa2': 0.0, 'rba4': 0.0, 'rpa3': 0.0, 'rba5': 0.0, 'rpa4': 0.0, 'rba6': 0.0, 'rpa5': 0.0, 'rpa6': 0.0}
# 
# Num Features: %d 10
# Selected Features: %s [False False False  True False False False False False False  True False
#  False False False  True  True  True  True  True  True  True  True]
# Feature Ranking: %s [14 13 10  1  7  6  5  4  3  2  1 11 12  9  8  1  1  1  1  1  1  1  1]
# 
# 
# =============================================================================


sss =model_selection.StratifiedShuffleSplit(n_splits=15, test_size=0.25, random_state=42)
sss.get_n_splits(X,y)
for train_index, test_index in sss.split(X,y):
    Xtrain, Xtest = X.iloc[train_index,:],X.iloc[test_index,:]
    ytrain, ytest = y.iloc[train_index],y.iloc[test_index]
    


rf=ensemble.RandomForestClassifier(n_estimators=300,criterion="gini",
                                   max_depth=9,max_leaf_nodes=100,
                                   min_samples_split=10,random_state=42)
rf.fit(Xtrain,ytrain)
prf=rf.predict(Xtest)
printresult(ytrain,npr)

best.fit(Xtrain,ytrain)
k=best.predict(Xtest)





nbmodel=naive_bayes.GaussianNB()
model.fit(Xtrain,ytrain)
npr=model.predict(Xtrain)   
printresult(ytest,npr) 
    
X.info()    
X.drop(["p1","p2","p3","p4","p5","p6","rba6","rba5","rba4","rpa6","rpa5","rpa4"],axis=1,inplace=True)    

modelstats1(Xtrain,Xtest,ytrain,ytest)
model_selection.GridSearchCV()

def modelstats1(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LR","DecisionTree","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(tree.DecisionTreeClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(5,17,2)]
            grid={"n_neighbors":k}
            grid_obj = GridSearchCV(estimator=model,param_grid=grid,scoring="recall")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    
    colnames=["MODELNAME","ACCURACY","PRECISION","RECALL","AUC"]
    return pd.DataFrame(stats,columns=colnames)
######################################
    


from xgboost import XGBClassifier


xgbmodel=XGBClassifier(max_depth=9,learning_rate=0.01,random_state=42)
xgbmodel.fit(Xtrain,ytrain)
prxgb=xgbmodel.predict(Xtest)
printresult(ytest,prxgb)









#####################################################################################################




















########roc curve
probs = xgbmodel.predict_proba(Xtest)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = metrics.roc_auc_score(ytest, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(ytest, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()

#########




















