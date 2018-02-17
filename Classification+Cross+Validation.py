
# coding: utf-8

# In[2]:


#import modules
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import itertools
from sklearn import linear_model
from sklearn import feature_selection
from sklearn import metrics

#import data
probeA = pd.read_csv('C:/Users/Joe/Downloads/probeA.csv',header=0)

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
def TNAregression(df):
    #Gives predicted TNA values using optimised ridge regression
    
    #Fixing data
    real_probeA = uncorrupt(df)
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
    
    #tna_predict is column of tna predictions using optimal regression model
    tna_predict = reg_new.predict(X_new)
    tna_predict = pd.DataFrame(data=tna_predict) #converting to pandas DF to export
    
    return tna_predict

#k-NN Algorithm
def kNN(df,k):
    #df is dataframe with last column the targets
    #k is number of nearest neighbours
    #returns the model
    copydf = df.copy()
    X=copydf.drop("class", 1)
    t=copydf["class"]
    neigh = KNeighborsClassifier(n_neighbors=k,metric='euclidean',weights='distance')
    neigh.fit(X, t)
    return neigh

#Decision Tree Algorithm
def DT(df,max_dep):
    #df is dataframe with last column the targets
    #max_dep is the max depth of the decision tree
    #returns the model
    copydf = df.copy()
    X=copydf.drop("class", 1)
    t=copydf["class"]
    clf = tree.DecisionTreeClassifier(max_depth=maxdep)
    clf = clf.fit(X, t)
    return clf

#Cross Validation
def CV(n,old_df,df,k,m,tna_predict,model_type):
    #old_df is a dataframe with last column as the targets
    #df is old_df but with some unimportant features dropped
    #n is the n in n-fold cross validation
    #maxdep is the maximum depth of the tree
    #k is the number of nearest neighbours OR max_depth if using decision tree
    #tna_predict is the regression prediction of tna values for our dataframe
    #model_type is either "knn" or "dt
    
    assert model_type == "knn" or model_type == "dt"
    
    #copy
    copydf = df.copy()
    
    #standardise tna_predict
    std_tna = (tna_predict - tna_predict.mean())/tna_predict.std()
    
    #creating the folds
    kf = cv.KFold(len(df),n)
    
    #area under curve
    auc=0
    
    for train,test in kf:
        
        traindf = copydf.iloc[train].copy()
        test_old_df = old_df.iloc[test].copy() #This exists for us to extracting data from for predictive tna features
        testdf  = copydf.iloc[test].copy()
        tna_pred = std_tna.iloc[test].copy()
              
        
        #Modify tna in test data for every tna feature in the test data
        if "tna" in testdf.columns:
            testdf["tna"] = tna_pred
        coltitles = ["c1", "c2", "c3", "m1", "m2", "m3", "n1", "n2", "n3", "p1", "p2", "p3","tna"]
        #iterate through all combinations
        for i in range(1,m,1):
            for pair in itertools.combinations_with_replacement(coltitles, i):
                #if the feature exists in the dataframe (i.e. it hasnt been dropped by featureselection)
                if "*".join(pair)+"*"+"tna" in testdf.columns:
                    testdf["*".join(pair)+"*"+"tna"] = tna_pred
                    for j in range(i):
                        testdf["*".join(pair)+"*"+"tna"] *= test_old_df[pair[j]] #this contains all features which testdf doesnt
        
        #Testdf without class to put in knn/dt model and get result of
        to_predict = testdf.drop("class",1)
    
        #define model and get the predictions
        if model_type == "knn":
            model = kNN(traindf,k)
        elif model_type == "dt":
            model = DT(traindf,k)
        else:
            print(ERROR)
            return
        predictions = model.predict_proba(to_predict)[:,1] #get prediction of class by regression model for each entry in traindf
        predictions = pd.DataFrame(data=predictions) #converting to pandas DF
    
        #update AUC
        auc += metrics.roc_auc_score(testdf["class"], predictions, average='macro', sample_weight=None)

    #return the average AUC
    return float(auc)/float(n)



#Feature Selection
def FeatSelect(df,k,m,tna_predict,thresh):
    #m is the dimension of feature expansion
    copydf = df.copy()
    old_df = copydf #This exists for us to extract values from later
    X=copydf.drop(["class"], 1)
    t=copydf["class"]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, t)
    #drop unimportant features for each treshold tested
    model = feature_selection.SelectFromModel(clf, threshold=thresh, prefit=True)
    copydf_new = model.transform(X)
    #Adding Column titles
    feature_names = np.array(X.columns)
    selected_features = feature_names[model.get_support()]
    copydf_new = pd.DataFrame(data= X, columns= selected_features)
    copydf_new["class"] = t
    #Test new model
    return [old_df,copydf_new]
    


# MAIN (we have chosen optimised values to iterate through, this can be edited)

tna_pred = TNAregression(probeA) # Predicted tna values using regression
model_type = "knn"

#Iterate through dimension of feature expansion, n
for n in [1]:
    
    #Fixing Data
    real_probeA = uncorrupt(probeA)
    FE_probeA = FE_with_tna(real_probeA,n)
    stdprobeA = standardisation_with_tna(FE_probeA)
    
    #Iterate through choice of regularisation constant
    for k in [32]:
        
        #Iterate through threshold for feature selection
        for thresh in [0.038]: 
            
            print("Model: "+model_type+", order n: "+str(n)+" and k/depth: "+str(k))
            [old_df,copydf_new] = FeatSelect(stdprobeA,k,n,tna_pred,thresh)
            print("Thresh: "+str(thresh)+" gives auc of "+str(CV(100,old_df,copydf_new,k,n,tna_pred,model_type)))

