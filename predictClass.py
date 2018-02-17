
# coding: utf-8

# In[30]:


#importing modules
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import itertools
from sklearn import linear_model
from sklearn import feature_selection

#import data
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

#Standardisation includng tna column
def standardisation_with_tna(df):
    #Removes the 'class' column, standardises the remaining data and puts class as a last column
    #Also standardises tna values
    copydf = df.copy()
    X=copydf.drop(["class"], 1)
    t=copydf["class"]
    for column in X:
        X[column] = (X[column] - X[column].mean())/X[column].std()
    X["class"] = t
    return X

#Standardisation excluding tna column
def standardisation_without_tna(df):
    #Standardises every column of the dataframe
    #We assume 'class' and 'TNA' do not exist in this dataframe
    X = df.copy()
    for column in X:
        X[column] = (X[column] - X[column].mean())/X[column].std()
    return X

#Feature Expansion up to n^th order monomials
def FE_without_tna(df, n):
    #Adds columns of all possible products of columns up to n^th order
    #Does not have products of tna column
    copydf = df.copy()
    coltitles = ["c1", "c2", "c3", "m1", "m2", "m3", "n1", "n2", "n3", "p1", "p2", "p3"]
    #include new column of products of features
    for i in range(2,n+1,1):
        for pair in itertools.combinations_with_replacement(coltitles, i):
            copydf["*".join(pair)] = 1
            for j in range(i):
                copydf["*".join(pair)] *= copydf[pair[j]]
    return copydf

#Feature Expansion up to n^th order monomials including tna
def FE_with_tna(df, n):
    #Adds columns of all possible products of columns up to n^th order, including tna
    copydf = df.copy()
    coltitles = ["c1", "c2", "c3", "m1", "m2", "m3", "n1", "n2", "n3", "p1", "p2", "p3","tna"]
    #include new column of products of features
    for i in range(2,n+1,1):
        for pair in itertools.combinations_with_replacement(coltitles, i):
            copydf["*".join(pair)] = 1
            for j in range(i):
                copydf["*".join(pair)] *= copydf[pair[j]]
    return copydf

#Using optimised ridge regression to predict tna values
def TNAregression(train_df,test_df):
    #Gives predicted TNA values using optimised ridge regression
    
    #Fixing training data
    real_probeA = uncorrupt(train_df)
    FE2_probeA = FE_without_tna(real_probeA,2)
    stdprobe2A = standardisation_without_tna(FE2_probeA)
    stdprobe2A["ones"] = 1
    copydf = stdprobe2A.copy()
    
    #Building regression model using known optimal hyper parameters
    X = copydf.drop(["tna"], 1)
    t = copydf["tna"]
    reg = linear_model.Ridge(alpha = 0.0016)
    reg.fit(X,t)
    
    #Feature selection
    model = feature_selection.SelectFromModel(reg, threshold=8.5, prefit=True)
    copydf_new = model.transform(X)
    
    #Adding Column titles for feature-selected dataframe
    feature_names = np.array(X.columns)
    selected_features = feature_names[model.get_support()]
    copydf_new = pd.DataFrame(data= X, columns= selected_features)
    copydf_new["tna"] = t
    
    #Rebuilding model for feature-selected dataframe
    X_new = copydf_new.drop(["tna"], 1)
    t_new = copydf_new["tna"]
    reg_new = linear_model.Ridge(alpha = 0.0016)
    reg_new.fit (X_new,t_new)
    
    #Fixing test data
    real_probeB = uncorrupt(test_df)
    FE2_probeB = FE_without_tna(real_probeB,2)
    stdprobe2B = standardisation_without_tna(FE2_probeB)
    stdprobe2B["ones"] = 1
    copydf_test = stdprobe2B[X_new.columns]
    
    #tna_predict is column of tna predictions using optimal regression model
    tna_predict = reg_new.predict(copydf_test)
    tna_predict = pd.DataFrame(data=tna_predict) #converting to pandas DF to export
    
    return tna_predict


#Feature Selection using decision tree
def FeatSelectDT(df,thresh):
    
    #thresh is the threshold hyper parameter
    
    copydf = df.copy()
    
    #Building decision tree model
    X=copydf.drop(["class"], 1)
    t=copydf["class"]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, t)
    
    #Drop unimportant features below the treshold hyper parameter
    model = feature_selection.SelectFromModel(clf, threshold=thresh, prefit=True)
    copydf_new = model.transform(X)
    
    #Adding Column titles
    feature_names = np.array(X.columns)
    selected_features = feature_names[model.get_support()]
    copydf_new = pd.DataFrame(data= X, columns= selected_features)
    copydf_new["class"] = t
    
    return copydf_new


#TRAINING MAIN

#Hyper-parameters determined by CV
n = 1 #dimension of feature expansion
k = 32 #k nearest neighbours
thresh = 0.038 #threshold for important features
        
#Fixing Data
real_probeA = uncorrupt(probeA)
FE_probeA = FE_with_tna(real_probeA,n)
stdprobeA = standardisation_with_tna(FE_probeA)

#Feature Selection
FS_df = FeatSelectDT(stdprobeA,thresh)

#Building model
model = KNeighborsClassifier(n_neighbors=k,metric='euclidean',weights='distance')
X = FS_df.drop("class",1)
t= FS_df["class"]
model.fit(X, t)


#TEST MAIN

#Fixing Data
real_probeB = uncorrupt(probeB)
tna_pred = TNAregression(probeA,probeB) # Predicted tna values using regression
real_probeB["tna"] = tna_pred #add the TNA prediction to our probeB dataframe
FE_probeB = FE_with_tna(real_probeB,n) #Feature expand with tna
stdprobeB = standardisation_without_tna(FE_probeB) #standardise the entire dataframe (even though FE_probeB contains TNA)
FS_stdprobeB = stdprobeB[X.columns] #get selected features

#Giving predictions
predictions = model.predict_proba(FS_stdprobeB)[:,1] #get prediction of class (probability) by regression model for each entry in stdprobeB
predictions = pd.DataFrame(data=predictions) #converting to pandas DF to export

#Output the probeB data through our model and export this as CSV
predictions.to_csv(path_or_buf='../classB.csv', index=False, header=False)

