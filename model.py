# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:48:55 2020

@author: Admin
"""
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('hiring.csv')
dataset['experience'].fillna(0, inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:,:3]

#converting words intointeger value
def convert_to_int(word):
    word_dict={0:0, 'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
               'six':6, 'seven':7, 'eight':8,'nine':9,'ten':10}
    return word_dict[word]

X['experience']= X['experience'].apply(lambda x : convert_to_int(x))

y=dataset.iloc[:,-1]

#Now training our model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

#fitting model with training data
regressor.fit(X,y)

#saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

#loading model to compare the results
model=pickle.load(open('model.pkl','rb'))
print(model.predict([2,9,6]))

