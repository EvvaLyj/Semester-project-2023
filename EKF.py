import math
import numpy as np

import matplotlib.pyplot as plt
import numpy.random as rd
from filterpy.kalman import ExtendedKalmanFilter

# 在0411的版本基础上
# 1. 修改state顺序(完成)
# 2. array输入（完成）
# 3. 使用calculate_phases函数来计算phases(完成，结果一致)
# 4. 检查Jacobian（看之钱大概0411左右内容）
# 5. 修改Jacobian(根据paper)（完成）
# 6. 修改f，lmb(完成)
# 7. 放入EXP1中并记录MSE（完成）
figsize=(3,2)

def plot_result(Phases,real_state,filter_result):
    plt.figure(figsize=figsize)
    plt.plot(range(1,len(Phases)), Phases[1:,0], label = 'Measurements')
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('phase [rad]',fontsize=14)

    plt.figure(figsize=figsize)
    plt.plot(range(1,len(real_state)), real_state[1:,0], label = 'Real statement',c='g' )
    plt.plot(range(1,len(filter_result)), np.array(filter_result)[1:,0], label = 'Extended Kalman Filter')
    plt.legend()
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('x-position [m]',fontsize=14)
    plt.show()
    
    plt.figure(figsize=figsize)
    plt.plot(range(1,len(real_state)), real_state[1:,1], label = 'Real statement',c='g' )
    plt.plot(range(1,len(filter_result)), np.array(filter_result)[1:,1], label = 'Extended Kalman Filter')
    plt.legend()
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('x-velocity [m]',fontsize=14)
    plt.show()
    
    plt.figure(figsize=figsize)
    plt.plot(range(1,len(real_state)), real_state[1:,2], label = 'Real statement',c='g' )
    plt.plot(range(1,len(filter_result)), np.array(filter_result)[1:,2], label = 'Extended Kalman Filter')
    plt.legend()
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('y-position [m]',fontsize=14)
    plt.show()

    
    plt.figure(figsize=figsize)
    plt.plot(range(1,len(real_state)), real_state[1:,3], label = 'Real statement',c='g' )
    plt.plot(range(1,len(filter_result)), np.array(filter_result)[1:,3], label = 'Extended Kalman Filter')
    plt.legend()
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('y-velocity [m]',fontsize=14)
    plt.show()


#不考虑modulo的实现
def HJacobian_at(x,lmb,arr):
    N=arr.shape[0]
    target=np.array([x[0,0],x[2,0]])
    J=np.zeros([N,4])
    for i in range(N):
        distancetoantenna = np.linalg.norm(target-arr[i])
        J1=(((target[0]-arr[i,0])*4*math.pi/lmb)/distancetoantenna)%(2*math.pi)
        J2=(((target[1]-arr[i,1])*4*math.pi/lmb)/distancetoantenna)%(2*math.pi)
        J[i,:]=np.array([J1,0,J2,0])
    return  J
def Hx(x,lmb,arr): 

    N=arr.shape[0]
    target=np.array([x[0,0],x[2,0]])
    phases=np.zeros([N,1])
    for i in range (N):
        distancetoantenna = np.linalg.norm(target-arr[i])
        phi_0 = 0
        phases[i,:] = (distancetoantenna * 4*math.pi/lmb+phi_0) % (2*math.pi)
    return phases

def EKF_v0(itr,dt,x0,P0,F,Q,R,arr,lmb,Phases):

    N=arr.shape[0]
    #EKF filter
    kf =ExtendedKalmanFilter(dim_x=4, dim_z=1) 
    #Initialize parameters
    kf.x = x0 
    kf.F = F 
    kf.P = P0 
    kf.R = R 
    kf.Q = Q
    filter_result=list()
    filter_result.append(x0)
    for i in range(1,itr):
        kf.predict()
        z = Phases[i,:].reshape([N,1])
        kf.update(z,HJacobian_at,Hx,args=(lmb,arr),hx_args=(lmb,arr))
        filter_result.append(kf.x)
    filter_result=np.array(filter_result)
    return filter_result
#implementation according to the paper
#1. 3D改成2D
#2. target起始点最好不要和q0重合，否则会除以0，导致NaN值出现 (x0改成[1,4,0,5])

def HJacobian_at_v1(x,lmb,arr,x0):
    
    N=arr.shape[0]
    target=np.array([x[0,0],x[2,0]])
    # target0=np.array([x0[0,0],x0[2,0]])
    J=np.zeros([N,4])
    for i in range(N):
        theta=np.arccos(0)
        theta_i0=np.arccos(0)
        phi=np.arctan2(target[1]-arr[0,1],target[0]-arr[0,0])
        phi_i0=np.arctan2(arr[i,1]-arr[0,1],arr[i,0]-arr[0,0])
        d= np.linalg.norm(target-arr[0])
        d_i0=np.linalg.norm(arr[0]-arr[i])
        delta_p_d=np.array([target[0]-arr[0,0],target[1]-arr[0,1]])/d
        delta_p_theta=np.array([np.cos(theta)*np.cos(phi)/d,np.sin(phi)*np.cos(theta)/d])
        delta_p_phi=np.array([-np.sin(phi)/(d*np.sin(theta)),np.cos(phi)/(d*np.sin(theta))]).reshape(2)
        g_i=np.sin(theta_i0)*np.sin(theta)*np.cos(phi_i0-phi)+np.cos(theta_i0)*np.cos(theta)
        delta_p_g_i=np.sin(theta_i0)*(np.cos(phi_i0-phi)*np.cos(theta)*delta_p_theta+np.sin(theta)*np.sin(phi_i0-phi)*delta_p_phi)-np.cos(theta_i0)*np.sin(theta)*delta_p_theta
        delta_p_f_i=-2*d_i0/d*(d_i0*delta_p_d/(d**2)+delta_p_g_i-g_i*delta_p_d/d)
        f_i=1+(d_i0/d)**2-2*d_i0/d*g_i
        delta_p_delta_d_i=delta_p_d*(np.sqrt(f_i)-1)+d*delta_p_f_i/(2*np.sqrt(f_i))
        
        delta_x_i=target[0]-arr[i,0]
        delta_y_i=target[1]-arr[i,1]
        d_i = np.linalg.norm(target-arr[i])
        # delta_p_d_i=np.array([delta_x_i*delta_p_delta_d_i[0],delta_y_i*delta_p_delta_d_i[1]])/d_i
        delta_p_d_i=np.array([delta_x_i,delta_y_i])/d_i
        
        J1=delta_p_d_i[0]
        J2=delta_p_d_i[1]
        J[i,:]=np.array([J1,0,J2,0])
    return  J


def EKF_v1(itr,dt,x0,P0,F,Q,R,arr,lmb,Phases):
    N=arr.shape[0]
    #EKF filter
    kf =ExtendedKalmanFilter(dim_x=4, dim_z=1) 
    #Initialize parameters
    kf.x = x0 
    kf.F = F 
    kf.P = P0 
    kf.R = R 
    kf.Q = Q
    filter_result=list()
    filter_result.append(x0)
    for i in range(1,itr):
        kf.predict()
        z = Phases[i,:].reshape([N,1])
        kf.update(z,HJacobian_at_v1,Hx,args=(lmb,arr,x0),hx_args=(lmb,arr))
        filter_result.append(kf.x)
    filter_result=np.array(filter_result)
    return filter_result

#phase difference as observations
def HJacobian_at_v2(x,lmb,arr,x0):
    
    N=arr.shape[0]
    target=np.array([x[0,0],x[2,0]])
    # target0=np.array([x0[0,0],x0[2,0]])
    J=np.zeros([N,4])
    for i in range(N):
        theta=np.arccos(0)
        theta_i0=np.arccos(0)
        phi=np.arctan2(target[1]-arr[0,1],target[0]-arr[0,0])
        phi_i0=np.arctan2(arr[i,1]-arr[0,1],arr[i,0]-arr[0,0])
        d= np.linalg.norm(target-arr[0])
        d_i0=np.linalg.norm(arr[0]-arr[i])
        delta_p_d=np.array([target[0]-arr[0,0],target[1]-arr[0,1]])/d
        delta_p_theta=np.array([np.cos(theta)*np.cos(phi)/d,np.sin(phi)*np.cos(theta)/d])
        delta_p_phi=np.array([-np.sin(phi)/(d*np.sin(theta)),np.cos(phi)/(d*np.sin(theta))]).reshape(2)
        g_i=np.sin(theta_i0)*np.sin(theta)*np.cos(phi_i0-phi)+np.cos(theta_i0)*np.cos(theta)
        delta_p_g_i=np.sin(theta_i0)*(np.cos(phi_i0-phi)*np.cos(theta)*delta_p_theta+np.sin(theta)*np.sin(phi_i0-phi)*delta_p_phi)-np.cos(theta_i0)*np.sin(theta)*delta_p_theta
        delta_p_f_i=-2*d_i0/d*(d_i0*delta_p_d/(d**2)+delta_p_g_i-g_i*delta_p_d/d)
        f_i=1+(d_i0/d)**2-2*d_i0/d*g_i
        delta_p_delta_d_i=delta_p_d*(np.sqrt(f_i)-1)+d*delta_p_f_i/(2*np.sqrt(f_i))
        
        # delta_x_i=target[0]-arr[i,0]
        # delta_y_i=target[1]-arr[i,1]
        # d_i = np.linalg.norm(target-arr[i])
        # delta_p_d_i=np.array([delta_x_i*delta_p_delta_d_i[0],delta_y_i*delta_p_delta_d_i[1]])/d_i
        
        J1=delta_p_delta_d_i[0]
        J2=delta_p_delta_d_i[1]
        J[i,:]=np.array([J1,0,J2,0])
    return  J

def Hx_v2(x,lmb,arr): 

    N=arr.shape[0]
    target=np.array([x[0,0],x[2,0]])
    phases=np.zeros([N,1])
    for i in range (N):
        distancetoantenna = np.linalg.norm(target-arr[i])
        distancetoref=np.linalg.norm(target-arr[0])
        phi_0 = 0
        phases[i,:] = ((distancetoantenna-distancetoref) * 4*math.pi/lmb+phi_0) % (2*math.pi)
    return phases

def EKF_v2(itr,dt,x0,P0,F,Q,R,arr,lmb,Phases):
    N=arr.shape[0]
    #EKF filter
    kf =ExtendedKalmanFilter(dim_x=4, dim_z=1) 
    #Initialize parameters
    kf.x = x0 
    kf.F = F 
    kf.P = P0 
    kf.R = R 
    kf.Q = Q
    filter_result=list()
    filter_result.append(x0)
    for i in range(1,itr):
        kf.predict()
        z = (Phases[i,:]-Phases[i,0]).reshape([N,1]) #phase diff
        kf.update(z,HJacobian_at_v2,Hx_v2,args=(lmb,arr,x0),hx_args=(lmb,arr))
        filter_result.append(kf.x)
    filter_result=np.array(filter_result)
    return filter_result