#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:44:51 2019

@author: delta
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
import category_encoders as ce

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
    
    
    
    
    
    
df.info()    
df=pd.read_csv("/home/delta/dataset/project/credit_default.csv")
colname={"ID":"id","EDUCATION":"edu","MARRIAGE":"mar","SEX":"sex","LIMIT_BAL":"lbal","PAY_0":"p1","PAY_2":"p2","PAY_3":"p3","PAY_4":"p4","PAY_5":"p5","PAY_6":"p6",
         "BILL_AMT1":"ba1","BILL_AMT2":"ba2","BILL_AMT3":"ba3","BILL_AMT4":"ba4","BILL_AMT5":"ba5","BILL_AMT6":"ba6",
         "PAY_AMT1":"pa1","PAY_AMT2":"pa2","PAY_AMT3":"pa3","PAY_AMT4":"pa4","PAY_AMT5":"pa5","PAY_AMT6":"pa6","default.payment.next.month":"dpay" }

df.rename(columns=colname,inplace=True)    
df.drop("id",axis=1,inplace=True)
##################
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
df.loc[change, 'p5'] = 0

#df.p5.value_counts()
change = (df.p6 == -1) | (df.p6 == -2) | (df.p6 == 0)
df.loc[change, 'p6'] = 0

#################

df['agecat'] = 0 #creates a column of 0
df.loc[((df['AGE'] >= 20) & (df['AGE'] < 30)) , 'agecat'] = 1
df.loc[((df['AGE'] >= 30) & (df['AGE'] < 40)) , 'agecat'] = 2
df.loc[((df['AGE'] >= 40) & (df['AGE'] < 50)) , 'agecat'] = 3
df.loc[((df['AGE'] >= 50) & (df['AGE'] < 60)) , 'agecat'] = 4
df.loc[((df['AGE'] >= 60) & (df['AGE'] < 70)) , 'agecat'] = 5
df.loc[((df['AGE'] >= 70) & (df['AGE'] < 81)) , 'agecat'] = 6

X['agecat1'] = 0 #creates a column of 0
X.loc[(X['agecat'] == 1) , 'agecat1'] = "young"
X.loc[(X['agecat'] == 2) , 'agecat1'] = "midyoung"
X.loc[(X['agecat'] == 3)  , 'agecat1'] = "mature"
X.loc[(X['agecat'] == 4)  , 'agecat1'] = "forty"
X.loc[(X['agecat'] == 5) , 'agecat1'] = "aged"
X.loc[(X['agecat']==6)  , 'agecat1'] = "old"

########
df.loc[1:1]
df.shape



X['sexcat'] = 0 #creates a column of 0
X.loc[(X['sex'] ==1), 'sexcat'] = "male"
X.loc[(X['sex'] ==2), 'sexcat'] = "female"
df.loc[((df['AGE'] >= 30) & (df['AGE'] < 40)) , 'agecat'] = 2
df.loc[((df['AGE'] >= 40) & (df['AGE'] < 50)) , 'agecat'] = 3
df.loc[((df['AGE'] >= 50) & (df['AGE'] < 60)) , 'agecat'] = 4
df.loc[((df['AGE'] >= 60) & (df['AGE'] < 70)) , 'agecat'] = 5
df.loc[((df['AGE'] >= 70) & (df['AGE'] < 81)) , 'agecat'] = 6



X['p6cat'] = 0 #creates a column of 0
X.loc[(X['p6'] ==0), 'p6cat'] = "payduly"
X.loc[(X['p6'] ==1), 'p6cat'] = "one"
X.loc[(X['p6'] ==2), 'p6cat'] = "two"
X.loc[(X['p6'] ==3), 'p6cat'] = "three"
X.loc[(X['p6'] ==4), 'p6cat'] = "four"
X.loc[(X['p6'] ==5), 'p6cat'] = "five"
X.loc[(X['p6'] ==6), 'p6cat'] = "six"
X.loc[(X['p6'] ==7), 'p6cat'] = "seven"
X.loc[(X['p6'] ==8), 'p6cat'] = "eight"





X=df[["p1","p2","p3","p4","p5","p6"]]


X.sample(10)

X1=X["sex"]
X.info()
X.p5.unique()

y=df[""]
X1=X.drop([],axis=1)

X1.sample(2)
Y=X.drop([],axis=1)
Y.sample(2)

X1=X.drop(["p1","p2","p3","p4","p5","p6"],axis=1)




encoder = ce.BinaryEncoder(cols=["p1cat","p2cat","p3cat","p4cat","p5cat","p6cat"])

Xenco=encoder.fit_transform(X1,Y)
Xenco.info()

Xenco.to_excel("pcat.xlsx",sheet_name='Sheet_name_1')

df=pd.read_excel("/home/delta/pcat.xlsx")
df.info()
df.sample()

df.info()
df.drop(["sex","edu","mar","AGE","p1","p2","p3","p4","p5","p6"],axis=1,inplace=True)
df.info()
pcat=pd.read_excel("/home/delta/pcat.xlsx")
pcat.info()
demo=pd.read_excel("/home/delta/output.xlsx")
demo.info()


cto=pcat.copy()
cto.info()
cto[["sex1","sex2","e1","e2","e3","m1","m2","m3","ag1","ag2","ag3","ag4"]]=demo[["sexcat_0","sexcat_1","ecat1","ecat2","ecad3","mcat1","mcat2","mcat3","agecat1","agecat2","agecat3","agecat4"]]
cto.info()
X=cto.copy()
y=df["dpay"]
depth=list(range(3,12))
grid = {'max_depth': np.arange(3, 20),
             'criterion' : ['gini','entropy'],
             'max_leaf_nodes': [5,10,20,30,60,50,100],
             'min_samples_split': [2, 5, 10, 20,30,40]}

gmodel=tree.DecisionTreeClassifier()
gmodel=model_selection.GridSearchCV(gmodel,grid,scoring="recall")
gmodel.fit(X,y)
best = gmodel.best_estimator_
print(best)
best.fit(X,y)
print(dict(zip(X.columns, best.feature_importances_)))

################################################################################
gmodel=tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=13,
            max_features=None, max_leaf_nodes=60,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=30,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')


gmodel.fit(Xtrain,ytrain)
t=gmodel.predict(Xtest)
printresult(ytest,t)

rmodel=ensemble.RandomForestClassifier(class_weight=None, criterion='entropy', max_depth=13,
            max_features=None, max_leaf_nodes=60,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=30,
            min_weight_fraction_leaf=0.0,  random_state=42,
            )

rmodel.fit(Xtrain,ytrain)
k=rmodel.predict(Xtest)
printresult(ytest,k)



































sss =model_selection.StratifiedShuffleSplit(n_splits=15, test_size=0.25, random_state=42)
sss.get_n_splits(X,y)
for train_index, test_index in sss.split(X,y):
    Xtrain, Xtest = X.iloc[train_index,:],X.iloc[test_index,:]
    ytrain, ytest = y.iloc[train_index],y.iloc[test_index]

best.fit(Xtrain,ytrain)
k=best.predict(Xtest)
printresult(ytest,k)

cto[["lbal","]]

df.info()

cto.info()


cto[["lbal","ba1","ba2","ba3","ba4","ba5","ba6","pa1","pa2","pa3","pa4","pa5","pa6"]]=df[["lbal","ba1","ba2","ba3","ba4","ba5","ba6","pa1","pa2","pa3","pa4","pa5","pa6"]



cto.info()

modelsnames


fx=cto.drop(["p1cat_0","p1cat_1","p1cat_2","p2cat_0","p2cat_1","p2cat_2",
             "p3cat_0","p3cat_1","p3cat_2","p4cat_0","p4cat_1","p4cat_2",
             "p5cat_0","p5cat_1","p5cat_2","p6cat_0","p6cat_1","p6cat_2",
             "e1","e2","e3","ba6","ba5","ba4","pa6","pa5","pa4"],axis=1)


fx.info()
X=fx.copy()
y=df.dpay

















