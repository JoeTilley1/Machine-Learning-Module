
# coding: utf-8

# In[20]:


#import modules
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import feature_selection
import itertools

#Import our data
probeA = pd.read_csv('../probeA.csv',header=0)
probeB = pd.read_csv('../probeB.csv',header=0)

#Uncorrupt our data
def uncorrupt(df):
    # we should look at compound c,m,n,p, and ensure that for each rows, for each compound then...
    #... its resolutions are in ascending order
    copydf=df.copy()
    for letter in ["c","m","n","p"]:
        old_l = copydf[[letter+"1",letter+"2",letter+"3"]]
        l = old_l.values
        l.sort(axis=1)
        l_df = pd.DataFrame(l,columns=old_l.columns)
        copydf[old_l.columns] = l_df
    return copydf

#Feature Expansion
def FE(df, n):
    #We add monomials of products of all compound/resolutions up to monomials of order n
    copydf = df.copy()
    coltitles = ["c1", "c2", "c3", "m1", "m2", "m3", "n1", "n2", "n3", "p1", "p2", "p3"]
    #include new column of products of features
    for i in range(2,n+1,1):
        for pair in itertools.combinations_with_replacement(coltitles, i):
            copydf["*".join(pair)] = 1
            for j in range(i):
                copydf["*".join(pair)] *= copydf[pair[j]]
    return copydf

def standardisationA(df):
    #Removes column class, standardises remaining data and puts tna at end
    #This will only standardise probeA
    copydf = df.copy()
    X=copydf.drop(["class","tna"], 1)
    t=copydf["tna"]
    for column in X:
        X[column] = (X[column] - X[column].mean())/X[column].std()
    X["tna"] = t
    return X

def standardisationB(df):
    #No need to remove columns class or tna because they don't exist in probeB
    #This will only standardise probeB
    X = df.copy()
    for column in X:
        X[column] = (X[column] - X[column].mean())/X[column].std()
    return X

def ridge(df,a):
    # Takes df and returns the ridge model
    # a is the regularisation constant
    copydf=df.copy()
    X = copydf.drop(["tna"], 1)
    t = copydf["tna"]
    reg = linear_model.Ridge(alpha = a)
    reg.fit (X,t)
    return reg

#Feature Selection with Ridge, returns new dataframe of just thought selected features 
def FeatSelectRidge(df,a,thresh):
    # a is the regularisation constant
    # thresh is the threshold with which we drop features with ridge coefficient below
    
    #fit model
    copydf = df.copy()
    X = copydf.drop(["tna"], 1)
    t = copydf["tna"]
    reg = linear_model.Ridge(alpha = a)
    reg.fit(X,t)
    
    #drop unimportant features
    model = feature_selection.SelectFromModel(reg, threshold=thresh, prefit=True)
    copydf_new = model.transform(X)
    
    #Adding Column titles
    feature_names = np.array(X.columns)
    selected_features = feature_names[model.get_support()]
    copydf_new = pd.DataFrame(data= X, columns= selected_features)
    copydf_new["tna"] = t
    
    return copydf_new


#TRAINING MAIN

#Hyper-parameters determined by LOOCV
n = 2 #dimension of feature expansion
a = 0.0016 #constant of regularisation
thresh = 8.5 #threshold for important features

#Fixing Data
real_probeA = uncorrupt(probeA)
FE_probeA = FE(real_probeA,n)
stdprobeA = standardisationA(FE_probeA)
stdprobeA["ones"] = 1

#Feature Selection
FS_stdprobeA = FeatSelectRidge(stdprobeA,a,thresh)

#Building model
model = ridge(FS_stdprobeA,a)
cols = FS_stdprobeA.columns
cols2 = cols.drop("tna")

#TESTING MAIN

#Fixing Data
real_probeB = uncorrupt(probeB)
FE_probeB = FE(real_probeB,n)
stdprobeB = standardisationB(FE_probeB)
stdprobeB["ones"] = 1
FS_stdprobeB = stdprobeB[cols2] #This gives the exact features required to fit the model

#Giving predictions
predictions = model.predict(FS_stdprobeB) #get prediction of class by regression model for each entry in stdprobeB
predictions = pd.DataFrame(data=predictions) #converting to pandas DF to export

#Output the probeB data through our model and export this as CSV
predictions.to_csv(path_or_buf='../tnaB.csv', index=False, header=False)

