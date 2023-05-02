import numpy as np
from filterpy.kalman import KalmanFilter
from ML_estimator import ML

def KF_ML(Length,dt,x0,P0,F,H,Q,R,lmb,ml_resol,std,arr,Phases):
    #dynamics: dt,x0,P0,F,H,Q,R,Length (length of a trajectory)
    #ML parameters: lmb,ml_resol,std,arr,Phases
    
    #filter
    kf =KalmanFilter(dim_x=4, dim_z=2) 
    #Initialize parameters
    kf.x = x0 #
    kf.P = P0
    kf.F = F
    kf.H = H
    kf.Q = Q
    kf.R = R
    

    #KF filter
    filter_result=list()
    filter_result.append(x0)
    z_k_list = [] #start at z_k1
    
    for i in range(1,Length):
        
        #predict
        kf.predict()#get x_k|x_k-1

        xrange=np.array([kf.x_prior[0,0]-3*std,kf.x_prior[0,0]+3*std])
        yrange=np.array([kf.x_prior[2,0]-3*std,kf.x_prior[2,0]+3*std])
        y_k=Phases[i,:]

        # ML
        ml=ML()
        num_grid,Loc_MLgrid=ml.ML_grid(xrange,yrange,y_k,arr,lmb,resolution=ml_resol,flag_prior=False,flag_plot=False)
        z_k_list.append(Loc_MLgrid.reshape(2,1)) #z_k
        z =Loc_MLgrid[0,:,np.newaxis]
    
        #update
        kf.update(z)
        filter_result.append(kf.x)
#     print(filter_result)
    filter_result=np.squeeze(np.array(filter_result))
    z_k_list = np.array(z_k_list)
    return z_k_list,filter_result
