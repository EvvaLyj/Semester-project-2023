"""
Evaluation.py

The evaluation metric for this project is MSE. 
"""

import numpy as np


def evaluate(true_target,estimate_target):
    
    # input size: L*M*N
    L = true_target.shape[0] # numner of tests(targets,trajectories)
    N = true_target.shape[1] # number of samples per test / Length of trajectories
    M = true_target.shape[2] # target dimension
    MSE = np.zeros(L)
    for i in range(L):
        error=true_target[i,:,:]-estimate_target[i,:,:]
        squared_error =np.linalg.norm(error,axis=1)**2 # taking norm of each row 
        MSE[i] = np.sum(squared_error)/(M*N)
    MSE_mean = np.mean(MSE)
    MSE_std = np.std(MSE)
    return MSE_mean,MSE_std


def todB(x):
    return 10*np.log(x)/np.log(10)



