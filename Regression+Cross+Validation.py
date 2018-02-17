
# coding: utf-8

# In[6]:


#import modules
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from sklearn import feature_selection
import itertools

#Import our data
probeA = pd.read_csv('../probeA.csv',header=0)

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

#Standardises Dataframe
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


# Ridge model
def ridge(df,a):
    # Takes df and returns ridge model
    # a is the regularisation parameter
    copydf=df.copy()
    X = copydf.drop(["tna"], 1)
    t = copydf["tna"]
    reg = linear_model.Ridge(alpha = a)
    reg.fit (X,t)
    return reg

# Lasso model
def lasso(df,a):
    # Takes df and returns lasso model
    # a is the regularisation parameter
    copydf=df.copy()
    X = copydf.drop(["tna"], 1)
    t = copydf["tna"]
    reg = linear_model.Lasso(alpha = a)
    reg.fit (X,t)
    return reg

# Cross Validation with choice of lasso/ridge model
def CV(n,df,a,model_type):
    #df is dataframe with last column as the targets
    #n is the n in n-fold cross validation
    #a the regularisation parameter
    #model_type should be either "ridge" or "lasso"
    
    assert model_type == "ridge" or model_type == "lasso"
    
    #creating the folds
    kf = cv.KFold(len(df),n,shuffle=True)
    err = 0
    for train,test in kf:
        
        traindf = df.iloc[train].copy()
        testdf  = df.iloc[test].copy()
        to_predict = testdf.drop("tna",1)
    
        #get the predictions
        if model_type == "ridge":
            model = ridge(traindf,a)
        elif model_type == "lasso":
            model = lasso(traindf,a)
        else:
            print(ERROR)
            return
        predictions = model.predict(to_predict) #get prediction of class by regression model for each entry in traindf
        predictions = pd.DataFrame(data=predictions) #converting to pandas DF
    
        #Update MSE error
        err += metrics.mean_squared_error(testdf["tna"], predictions, sample_weight=None, multioutput='uniform_average')
    
    #Return MSE
    return float(err)/float(n)


# Feature Selection to give new dataframe
def FeatSelect(df,a,thresh,model_type):
    # a is the regularisation constant
    # thresh is the threshold with which we drop features with ridge coefficient below
    #model_type should be either "ridge" or "lasso"
    
    assert model_type == "ridge" or model_type == "lasso"
    
    #fit model
    copydf = df.copy()
    X = copydf.drop(["tna"], 1)
    t = copydf["tna"]
    if model_type == "ridge":
        reg = linear_model.Ridge(alpha = a)
    elif model_type == "lasso":
        reg = linear_model.Lasso(alpha = a)
    else:
        print(ERROR)
        return 
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
        
    
        
# MAIN (we have chosen optimised values to iterate through, this can be edited)

#Iterate through dimension of feature expansion, n
for n in [2]:
    
    #Type of model
    model_type = "ridge"

    #Fixing Data
    real_probeA = uncorrupt(probeA)
    FE_probeA = FE(real_probeA,n)
    stdprobeA = standardisationA(FE_probeA)
    stdprobeA["ones"] = 1
    
    #Iterate through choice of regularisation constant
    for a in [0.0016]:
        
        #Iterate through threshold for feature selection
        for thresh in [8.5]:
            
            print("Model: "+model_type+", order n: "+str(n)+", a="+str(a))
            FS_df = FeatSelect(stdprobeA,a,thresh,model_type)
            print("Thresh: "+str(thresh)+" gives error of "+str(CV(1000,FS_df,a,model_type)))

