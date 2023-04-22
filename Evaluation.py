import numpy as np

def MSE_error(y_true,y_predict): #
    #y_true N*2
    square = np.linalg.norm(y_true-y_predict,axis=0)
    MSE = np.sum(square) / len(y_true)
    return MSE

def todB(x):
    return 10*np.log(x)/np.log(10)


def evaluate(true_target,estimate_target):
    squared_loss=((true_target - estimate_target)**2).mean()
    std=np.std(estimate_target,axis=1).mean(axis=0)
    return todB(squared_loss),std