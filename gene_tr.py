import numpy as np
import math
from numpy.random import randn 

def array(num_elements,dim,x_start):
    lin_array=np.zeros((num_elements,2))
    lin_array[:,0]=[i for i in np.linspace(x_start,x_start+dim,num_elements)]#y_axis
    lin_array[:,1]=[ 0 for _ in range(num_elements)]#x_axis
    return lin_array

def Calcphases(states,array,lmb,Phase_0,phase_noise_std):
    # given the the locations of trajectories, calculate the phases 
    # N: number of antenna elements
    # length: Length of trajectory
    N = array.shape[0]
    length=len(states)
    phases = np.zeros((length, N))
    
    if length > 1:
         for i in range(length):
            for j in range(N):
                distancetoantenna = np.linalg.norm(states[i]-array[j])
                phases[i, j] = (distancetoantenna * 4*math.pi/lmb + Phase_0+randn(1)*phase_noise_std) % (2*math.pi)
    if length == 1:
        phases = np.zeros(1,N)
        for j in range(N):
            distancetoantenna = np.linalg.norm(states-array[j].numpy())
            phases[0,j] = (distancetoantenna * 4*math.pi/lmb + Phase_0+randn(1)*phase_noise_std) % (2*math.pi)
    
    return phases


def GenerateTraj(Length,X0,F,Q):
    #state x=[px,vx,py,vy]
    #dynamics: Length, dt, X0, H,F,Q,R
    
    real_states = []
    position_measurements = []
    #process(motion) model 
    #observation model
    x = X0
    for i in range(Length):
        real_states.append(x)
        x = np.dot(F,x)+np.random.multivariate_normal(mean=(0,0,0,0),cov=Q).reshape(4,1)
        # position_measurements.append(np.array(np.dot(H,x)+np.random.multivariate_normal(mean=(0,0),cov=R).reshape(2,1)))
    # position_measurements = np.array(position_measurements).squeeze()
    real_states = np.array(real_states)


    return real_states

def GeneratePhasesPositionsforTrajectories(N_tr,Length,arr,lmb,phase_offset,real_states,position_noise_std,phase_noise_std):
    N=arr.shape[0] #number of antenna elements
    Phases=np.empty([N_tr,Length,N])
    noisy_positions = np.empty([N_tr,Length,2])
    for i in range(N_tr):
        real_positions_i=np.array([real_states[i,:,0],real_states[i,:,2]]).T
        noisy_positions_i=real_positions_i+randn(Length, 2)*lmb*position_noise_std
        Phases[i,:,:]=Calcphases(noisy_positions_i,arr,lmb,phase_offset,phase_noise_std)
        noisy_positions[i,:,:]=noisy_positions_i
    return Phases,noisy_positions
