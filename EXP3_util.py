from Basic import *
from numpy.random import randn 
import matplotlib.pyplot as plt
import math
figsize=(8,6)

def EXP3_phase_generate(N,lmb,d,targets,phase_noise_std,phi_0): #directly adding noise to the true phase
    # Array
    arr_dim=d*(N-1) # dimension of the array
    arr=array(N,arr_dim,0) # array elements location

    #Prepare arrays for results
    num_sam=5
    true_targets=np.empty([len(targets),num_sam,2])
    phase_measurements=np.empty([len(targets),num_sam,N])
    for i in range (len(targets)):
        for j in range(num_sam):
            true_targets[i,j]=targets[i]
            dist=np.array([np.linalg.norm(arr[n]-true_targets[i,j]) for n in range(N)]) #distance between the elements and target
            phase_measurements[i,j]=(4*np.pi/lmb*dist+phi_0+randn(N)*phase_noise_std)%(2*np.pi)
    return true_targets,phase_measurements

def EXP3_position_noise_to_phase_generate(N,lmb,d,targets,position_noise_std,phi_0): #adding noise to posititon, then generate a noise-free phase
    # noise std is position noise
    # Array
    arr_dim=d*(N-1) # dimension of the array
    arr=array(N,arr_dim,0) # array elements location

    #Prepare arrays for results
    num_sam=5
    true_targets=np.empty([len(targets),num_sam,2])
    noisy_targets=np.empty([len(targets),num_sam,2])
    phase_measurements=np.empty([len(targets),num_sam,N])
    for i in range (len(targets)):
        for j in range(num_sam):
            true_targets[i,j]=targets[i]
            noisy_targets[i,j]=true_targets[i,j]+lmb*randn(2)*position_noise_std
            dist=np.array([np.linalg.norm(arr[n]-noisy_targets[i,j]) for n in range(N)])#distance between the elements and target
            phase_measurements[i,j]=(4*np.pi/lmb*dist+phi_0)%(2*np.pi)
    return true_targets,phase_measurements

def Synthetic_prior(target,num_sample,prior_sigma):
    K=len(target)
    prior_means=np.zeros((K,num_sample,2))
    for i in range(K):
        for j in range(num_sample):
            prior_means[i,j,:]=np.random.multivariate_normal(mean=target[i],cov=prior_sigma**2*np.eye(2)).reshape(2)
    return prior_means

def plot_result_EXP3_1(noise_scale_dB,ML_MSE_mean_dB,ML_MSE_mean_prior_dB,MSE_floor_dB,ml_resol,lmb):#fix prior_sigma
    plt.figure(figsize=figsize)
    color=['b','g','purple','LightBLue']
    for j in range(len(ml_resol)):
        plt.plot(noise_scale_dB,ML_MSE_mean_dB[:,j],'o-',label=r'ML $s_{grid}=$'rf'{ml_resol[j]/lmb}$\lambda$',c=color[j])
        plt.plot(noise_scale_dB,ML_MSE_mean_prior_dB[:,j],'^-.',label=r'ML+prior $s_{grid}=$'rf'{ml_resol[j]/lmb}$\lambda$',c=color[j])
    plt.plot(noise_scale_dB,MSE_floor_dB,'d--',label=r'noise level',c='red')
    # plt.legend(bbox_to_anchor=(0.95, 1.2),ncol=3)
    plt.legend()

    plt.xlabel(r'$σ^{-2}$ [dB]')
    plt.ylabel(r'MSE $\hat{u}$ [dB]')
    plt.ylim([-30,10])
    plt.grid()
    
def plot_result_EXP3_2(noise_scale_dB,ML_MSE_mean_dB,ML_MSE_mean_prior_dB,MSE_floor_dB,prior_sigma,lmb,fixed_ml_resol):#fix ml_resol
    plt.figure(figsize=figsize)
    color=['b','g','purple','LightBLue','orange']
    plt.plot(noise_scale_dB,ML_MSE_mean_dB[:,0],'o-',label=r'ML ',c='black')
    for j in range(len(prior_sigma)):
        if(prior_sigma[j]<=2*fixed_ml_resol and prior_sigma[j]>=0.5*fixed_ml_resol):
            plt.plot(noise_scale_dB,ML_MSE_mean_prior_dB[:,j],'^-.',label=r'ML+prior $\sigma_{prior}=$'rf'{prior_sigma[j,0]/lmb}$\lambda$',c=color[j])
    plt.plot(noise_scale_dB,MSE_floor_dB,'d--',label=r'noise level',c='red')
    # plt.legend(bbox_to_anchor=(0.95, 1.2),ncol=3)
    plt.legend()

    plt.xlabel(r'$σ^{-2}$ [dB]')
    plt.ylabel(r'MSE $\hat{u}$ [dB]')
    plt.ylim([-30,10])
    plt.grid()

def Position_dB_to_phase_dB(Position_noise_dB,lmb):
    print(f'Position_noise_dB:{Position_noise_dB}')
    position_noise_std=Noise_dB_to_std(Position_noise_dB); print(f'position_noise_std:{position_noise_std}')
    phase_nosie_std=(lmb*position_noise_std)*math.pi*4/lmb; print(f'phase_nosie_std:{phase_nosie_std}')
    phase_noise_dB=-20*np.log10(phase_nosie_std); print(f'phase_noise_dB:{phase_noise_dB}')
    return phase_noise_dB

#验证过，这样和对noise_dB取负一样
# MSE_level=np.empty(len(position_noise_std))
# num_sam=5
# for k in range(len(position_noise_std)):
#     true_targets=np.empty([len(targets),num_sam,2])
#     noisy_targets=np.empty([len(targets),num_sam,2])
#     for i in range (len(targets)):
#         for j in range(num_sam):
#             true_targets[i,j]=targets[i]
#             noisy_targets[i,j]=true_targets[i,j]+lmb*randn(2)*position_noise_std[k]
#     MSE_level[k],_=evaluate(true_targets,noisy_targets)
# print(MSE_level)
# MSE_level_scale=MSE_level/(lmb**2)
# MSE_level_dB=todB(MSE_level_scale)
