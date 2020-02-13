# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:42:06 2020

@author: Khare
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:15:05 2020

@author: Khare
"""

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
encoder = LabelEncoder()
plt.style.use('ggplot')
        
#returns table with missing data info
def null_info(df):
    missing_table = pd.DataFrame(df.isnull().sum().sort_values(ascending = False), columns = ['total'])
    missing_table['percent'] = missing_table['total'].apply(lambda x:round(x/len(df)*100,2))
    return missing_table

#function to plot distribution and probability plot
def show_dist(label):
    ax =sns.distplot(label, fit=st.norm)
    ax.set_title("Skewness: {}".format(label.skew()))
    plt.figure()
    st.probplot(label, plot = plt)
    plt.show()

#visualize null value 
def visualize_nulls(df):
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
    
#function to fill missing values for categories
def handle_categorical(multcolumns, df):
    copy = pd.DataFrame()
    i = 0
    for field in multcolumns:
        temp = df[field].fillna(df[field].mode()[0])
        df.drop([field], axis = 1, inplace = True)
        
        if i == 0:
            copy = temp.copy()
        else:
            copy = pd.concat([copy,temp], axis=1)
            
        i += 1
    
    copy = pd.concat([df, copy], axis=1)
    return copy

#function to fill numeric values 
def handle_numeric(multcolumns, df):
    copy = pd.DataFrame()
    i = 0
    
    for field in multcolumns:
        temp = df[field].fillna(df[field].mean())
        df.drop([field], axis=1, inplace = True)
        
        if i == 0:
            copy = temp.copy()
        else:
            copy = pd.concat([copy, temp], axis=1)
        
        i += 1
    copy = pd.concat([df, copy], axis = 1)
    return copy
def label_Encoding(df, multcolumns):
    count = 0
    
    #iterate through the columns
    for col in multcolumns:
        encoder.fit(df[col])
        df[col] = encoder.transform(df[col])
        
        #keep track of columns encoded
        count += 1
    print("Label encoded columns: {}".format(count))
        
def one_hot_encoding(df,multcolumns):
    copy = pd.DataFrame()
    i = 0
    for field in multcolumns:
        tmp = pd.get_dummies(df[field], drop_first=True)
        df.drop([field], axis = 1, inplace = True)
        if i == 0:
            copy = tmp.copy()
        else:
            copy = pd.concat([copy, tmp], axis = 1)
        i += 1
    
    copy = pd.concat([df, copy], axis = 1)
    return copy

def rmsle(y_true, y_pred):
    diffs = np.log(y_true+1)-np.log(y_pred+1)
    squares = np.power(diffs, 2)
    err = np.sqrt(np.mean(squares))
    return err

def evaluate_model(y_pred, y_test):
    r2_measure = r2_score(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    rmsle_err = rmsle(y_pred, y_test)
    return r2_measure, RMSE, rmsle_err

def fit_models(X_train, y_train, X_test, y_test, models):
    #list to store performance of the models
    df_eval = pd.DataFrame()
    for key, value in models.items():
        print('Fitting: \t{}'.format(key))
        value.fit(np.array(X_train), np.array(y_train))
        y_pred = value.predict(np.array(X_test))
        r2, rmse, rmsle_err = evaluate_model(y_pred, y_test)
        print('Done!')
        df_temp = pd.DataFrame({'model':[key],'rmse':[rmse], 'r2':[r2], 'rmsle':[rmsle_err]})
        df_eval = df_eval.append(df_temp)
    
    print('=== Fitting Completed ! ====')
    return df_eval
