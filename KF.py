"""
KF.py

Here are 3 Kalman filters, which are 
1. KF (using position measurements)
2. KF with ML encoder (using phase measurements)
3. KF with ML encoder (using phase measurements), and for each time-step of the trajectory, run multiple iterations of Encoder + KF 

"""
import numpy as np
from filterpy.kalman import KalmanFilter
from ML_estimator import ML

def KF(Length,dt,x0,P0,F,H,Q,R,Positions):
    #dynamics: dt, x0, P0, F, H, Q, R, Length (length of a trajectory)
    #ML parameters: lmb, ml_resol, std, arr, Phases
    
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
        z=Positions[i]
        
        #predict
        kf.predict()
        #update
        kf.update(z)
        filter_result.append(kf.x)
#     print(filter_result)
    filter_result=np.squeeze(np.array(filter_result))
    return filter_result

#---------------------------------------------
def KF_ML(Length,dt,x0,P0,F,H,Q,R,lmb,ml_resol,std,arr,Phases,search_factor):
    #dynamics: dt, x0, P0, F, H, Q, R, Length (length of a trajectory)
    #ML parameters: lmb, ml_resol, std, arr, Phases, search factor
    
    #filter
    kf =KalmanFilter(dim_x=4, dim_z=2) 
    #Initialize parameters
    kf.x = x0 #inital state
    kf.P = P0 #initial P
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

        # ML
        xrange=np.array([kf.x_prior[0,0]-search_factor*std,kf.x_prior[0,0]+search_factor*std])
        yrange=np.array([kf.x_prior[2,0]-search_factor*std,kf.x_prior[2,0]+search_factor*std])
       
        y_k=Phases[i,:]
        ml=ML()
        num_grid,Loc_MLgrid=ml.ML_grid(xrange,yrange,y_k,arr,lmb,resolution=ml_resol,flag = False,flag_plot=False)
        z_k_list.append(Loc_MLgrid.reshape(2,1)) #z_k
        z =Loc_MLgrid.reshape(2,1)
    
        #update
        kf.update(z)
        filter_result.append(kf.x)
#     print(filter_result)
    filter_result=np.squeeze(np.array(filter_result))
    z_k_list = np.array(z_k_list)
    return z_k_list,filter_result

#---------------------------------------------
def KF_ML_ite(Length,dt,x0,P0,F,H,Q,R,lmb,ml_resol,std,arr,Phases,search_factor):
    #dynamics: dt, x0, P0, F, H, Q, R, Length (length of a trajectory)
    #ML parameters: lmb, ml_resol, std, arr, Phases, search_factor
    
    # For each time-step of the trajectory, run multiple iterations of Encoder + KF
    ite_num = 3

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
    filter_result_all=list()
    filter_result.append(x0)
    filter_result_all.append(x0)
    z_k_list = [] #start at z_k1
    
    for i in range(1,Length):
        for j in range(ite_num):
            if(j==0):
                #predict
                kf.predict()#get x_k|x_k-1
                
                # ML
                xrange=np.array([kf.x_prior[0,0]-search_factor*std,kf.x_prior[0,0]+search_factor*std])
                yrange=np.array([kf.x_prior[2,0]-search_factor*std,kf.x_prior[2,0]+search_factor*std])
            
                y_k=Phases[i,:]
                ml=ML()
                num_grid,Loc_MLgrid=ml.ML_grid(xrange,yrange,y_k,arr,lmb,resolution=ml_resol,flag = False,flag_plot=False)
                z = Loc_MLgrid.reshape(2,1)
                print(f'ite{j},z={z}')
                #update
                kf.update(z)
                post_x=kf.x

            else:
                # #predict
                # kf.predict()#get x_k|x_k-1
                
                # ML + grid search
                xrange=np.array([post_x[0,0]-search_factor*std,post_x[0,0]+search_factor*std])
                yrange=np.array([post_x[2,0]-search_factor*std,post_x[2,0]+search_factor*std])
            
                y_k=Phases[i,:]
                ml=ML()
                num_grid,Loc_MLgrid=ml.ML_grid(xrange,yrange,y_k,arr,lmb,resolution=ml_resol,flag = False,flag_plot=False)
                z = Loc_MLgrid.reshape(2,1)
                print(f'ite{j},z={z}')
                #update
                kf.update(z)
                post_x=kf.x
            
            #store
            filter_result_all.append(kf.x)
            if(j==ite_num-1):
                z_k_list.append(Loc_MLgrid.reshape(2,1)) #z_k
                filter_result.append(kf.x)
#     print(filter_result)
    filter_result=np.squeeze(np.array(filter_result))
    z_k_list = np.array(z_k_list)
    return z_k_list,filter_result,filter_result_all