# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:46:48 2020

@author: Asger

Sorry to whoever has to read this.. Didn't have time for cleaning it up'
"""

import numpy as np
import pandas as pd
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.tree
from toolbox_02450 import rlr_validate
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
import scipy.stats

# Read relevant columns from data and filter out rows with nan values.
data = pd.read_csv(r'covid-data.csv')
data = data.drop_duplicates(subset='iso_code', keep = 'last')

# We wish to predict deaths per million based on pop density, median age, develop index, diabetes prevalence
data = data[['total_deaths_per_million', 'population_density', 'median_age', 'human_development_index', 'diabetes_prevalence']]
data = data.dropna()

X = data[['population_density', 'median_age', 'human_development_index', 'diabetes_prevalence']]
X = X.reset_index(drop=True)
y = data[['total_deaths_per_million']].squeeze()
y = y.reset_index(drop=True)
X = X.to_numpy()
y = y.to_numpy()

attributeNames = data[['population_density', 'median_age', 'human_development_index', 'diabetes_prevalence']].columns.to_numpy()

N, M = X.shape
          
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = np.insert(attributeNames, 0, 'Offset', axis=0)
M = M+1

#Lambda values
lambdas = np.power(10.,range(-5,9))

#Cross validation
K = 5
CV = model_selection.KFold(K, shuffle=True)
w = np.empty((M,K,len(lambdas)))
train_error = np.empty((K,len(lambdas)))
test_error = np.empty((K,len(lambdas)))
f = 0

for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
        
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
        
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
        
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    for l in range(0,len(lambdas)):
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
        test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)

    f=f+1

opt_val_err = np.min(np.mean(test_error,axis=0))
opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
train_err_vs_lambda = np.mean(train_error,axis=0)
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))


figure(figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') 
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
legend(attributeNames[1:], loc='best')
        
subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()
show()

# Parameters for neural network classifier
n_hidden_units = [1,5,10,15,20]      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

E_test_baseline = np.empty(K)
E_test_rlr = np.empty(K)
E_test_ANN = np.empty(K)
lambda_stars = np.empty(K)
h_stars = np.empty(K)

rlr_test_errors_vs_lambda = np.empty((lambdas.shape[0], K))
baseline_test_errors = np.empty(K)
ANN_test_errors = np.empty((K,len(n_hidden_units)))

alpha = 0.05
CI_BASE_VS_RLR = np.empty(K)
CI_BASE_VS_ANN = np.empty(K)
CI_RLR_VS_ANN = np.empty(K)
P_BASE_VS_RLR = np.empty(K)
P_BASE_VS_ANN = np.empty(K)
P_RLR_VS_ANN = np.empty(K)
BASELINE_ERRORS_SQUARED = 0

k=0
############ Part B ################
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    errors = []
    
    f = 0
    for train_index_inner, test_index_inner in CV.split(X_train, y_train):
        print('\nCrossvalidation fold: {0}/{1}'.format(f,K))  
        
        X_train_inner = X[train_index_inner]
        y_train_inner = y[train_index_inner]
        X_test_inner = X[test_index_inner]
        y_test_inner = y[test_index_inner]
        
        X_train_inner_ANN = torch.Tensor(X[train_index_inner,:])
        y_train_inner_ANN = torch.Tensor(y[train_index_inner])
        X_test_inner_ANN = torch.Tensor(X[test_index_inner,:])
        y_test_inner_ANN = torch.Tensor(y[test_index_inner])
        
        baseline_model = np.mean(y_train_inner)
        baseline_test_errors[f] = np.power(y_test_inner - baseline_model,2).mean(axis=0)
        
        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train_inner[:, 1:], 0)
        sigma = np.std(X_train_inner[:, 1:], 0)
        
        X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu) / sigma
        X_test_inner[:, 1:] = (X_test_inner[:, 1:] - mu) / sigma
        
        # precompute terms
        Xty = X_train_inner.T @ y_train_inner
        XtX = X_train_inner.T @ X_train_inner
    
        for l in range(0,len(lambdas)):

            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train_inner-X_train_inner @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test_inner-X_test_inner @ w[:,f,l].T,2).mean(axis=0)
            

        for i in range(0,len(n_hidden_units)):
            model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units[i]), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units[i], 1), # n_hidden_units to 1 output neuron
                    )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_inner_ANN,
                                                       y=y_train_inner_ANN,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
            ANN_test_errors[f,i] = final_loss
        
        f = f+1
        
    E_gen_baseline = ((y_test_inner.shape[0]/y_test.shape[0])*baseline_test_errors).mean(axis=0)
    E_gen_rlr = ((y_test_inner.shape[0]/y_test.shape[0])*test_error).mean(axis=0)
    E_gen_ANN = ((y_test_inner.shape[0]/y_test.shape[0])*ANN_test_errors).mean(axis=0)
    
    baseline_model = np.mean(y_train)
    BASELINE_ERRORS_SQUARED = np.power(y_test - baseline_model,2)
    E_test_baseline[k] = BASELINE_ERRORS_SQUARED.mean(axis=0)
    
    mu = np.mean(X_test[:, 1:], 0)
    sigma = np.std(X_test[:, 1:], 0)
        
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    # precompute terms
    Xty = X_train_inner.T @ y_train_inner
    XtX = X_train_inner.T @ X_train_inner
    
    lambdaI = lambdas[np.argmin(E_gen_rlr)] * np.eye(M)
    lambdaI[0,0] = 0 # remove bias regularization
    w_star = np.empty(M)
    w_star = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Evaluate training and test performance
    RLR_ERRORS_SQUARED = np.power(y_test-X_test @ w_star,2)
    E_test_rlr[k] = RLR_ERRORS_SQUARED.mean(axis=0)
    
    X_test_ANN = torch.Tensor(X[test_index,:])
    X_test_ANN2 = X_test_ANN
    y_test_ANN = torch.Tensor(y[test_index])
    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units[np.argmin(E_gen_ANN)]), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units[np.argmin(E_gen_ANN)], 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
    loss_fn = torch.nn.MSELoss() 
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_test_ANN,
                                                       y=y_test_ANN,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    ANN_y_est = net(X_test_ANN2)
    ANN_y_est = ANN_y_est.data.numpy().squeeze()
    
    ANN_ERRORS_SQUARED = np.power(ANN_y_est - y_test,2)
    
    E_test_ANN[k] = ANN_ERRORS_SQUARED.mean(axis = 0)
    
    lambda_stars[k] = np.argmin(E_gen_rlr)
    h_stars[k] = np.argmin(E_gen_ANN)
    
    z1 = BASELINE_ERRORS_SQUARED - RLR_ERRORS_SQUARED
    ci1 = scipy.stats.t.interval(1-alpha, z1.shape[0] - 1, loc=np.mean(z1), scale = scipy.stats.sem(z1))
    P1 = scipy.stats.t.cdf( -np.abs( np.mean(z1) )/scipy.stats.sem(z1), df=z1.shape[0]-1)
    
    z2 = BASELINE_ERRORS_SQUARED - ANN_ERRORS_SQUARED
    ci2 = scipy.stats.t.interval(1-alpha, z2.shape[0] - 1, loc=np.mean(z2), scale = scipy.stats.sem(z2))
    P2 = scipy.stats.t.cdf( -np.abs( np.mean(z2) )/scipy.stats.sem(z2), df=z2.shape[0]-1)
    
    z3 = RLR_ERRORS_SQUARED - ANN_ERRORS_SQUARED
    ci3 = scipy.stats.t.interval(1-alpha, z3.shape[0] - 1, loc=np.mean(z3), scale = scipy.stats.sem(z3))
    P3 = scipy.stats.t.cdf( -np.abs( np.mean(z3) )/scipy.stats.sem(z3), df=z3.shape[0]-1)

    k = k+1







        
        
        
