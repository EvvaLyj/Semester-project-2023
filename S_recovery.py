import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def Unwrap(phase):
    phase_u=np.zeros(len(phase))
    for i in range(len(phase)):
        if (i==0):
            phase_u[i]=phase[i]
        else:
            phase_u[i]=phase[i]-2*np.pi*np.floor(0.5+(phase[i]-phase_u[i-1])/(2*np.pi))
    return phase_u

class sparse_recovery:
    def SSR_1d(self,lamb,N,q,q0,xrange,yrange,ssr_resol,y):
        
        num_x=int(np.floor((xrange[1]-xrange[0])/ssr_resol)+1)
        num_y=int(np.floor((yrange[1]-yrange[0])/ssr_resol)+1)
        
        #cadidate locations
        grid_x=np.linspace(xrange[0],xrange[1],num_x)
        grid_y=np.linspace(yrange[0],yrange[1],num_y)
        p_candidate=np.empty([num_x,num_y,2])
        for i in range(num_x):
            for j in range(num_y):
                p_candidate[i,j,0]=grid_x[i]
                p_candidate[i,j,1]=grid_y[j]
    #     print(p_candidate.shape)

        #Iterate over all y and save the best estimation at each y
        best_each_y=np.empty([num_y,2])
        for k in range(num_y):
            p_candidate_k=p_candidate[:,k]
            A=np.empty([N,num_x])
            for i in range(N):
                for j in range(num_x):
                    A[i,j]=4*np.pi/lamb*(np.linalg.norm(p_candidate_k[j]-q[i])-np.linalg.norm(p_candidate_k[j]-q0[i]))
        #     optimization y=Ax     
            n_nonzero_coefs=1
            omp = make_pipeline(StandardScaler(with_mean=False), OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs))
            omp.fit(A, y)
            coef=omp['orthogonalmatchingpursuit'].coef_
        #     print(coef)
            target_inx=np.nonzero(coef)
            loc_sparse_k=p_candidate_k[target_inx].squeeze()
    #         print(f"Estimated location at y={p_candidate[1,k][1]} is{loc_sparse_k}")
            best_each_y[k]=loc_sparse_k

        #Iterate over all best estimations at each y
        A=np.empty([N,len(best_each_y)])
        for i in range(N):
            for j in range(len(best_each_y)):
                A[i,j]=4*np.pi/lamb*(np.linalg.norm(best_each_y[j]-q[i])-np.linalg.norm(best_each_y[j]-q0[i]))
        # Optimization y=Ax       
        n_nonzero_coefs=1
        omp = make_pipeline(StandardScaler(with_mean=False), OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs))
        omp.fit(A, y)
        coef=omp['orthogonalmatchingpursuit'].coef_
        # print(coef)
        target_inx=np.nonzero(coef)
        loc_sparse=best_each_y[target_inx].squeeze()
#         print(f"SSR estimation is{loc_sparse}")

        return p_candidate,best_each_y,loc_sparse
    
    def plotting(self,size,f,p_candidate,best_each_y,loc_sparse,targets,arr,xrange,yrange,ssr_resol):
        plt.figure()
        #Array
        plt.scatter(arr[:,0],arr[:,1],color='black')
        #True target locations
        for i in range(len(targets)):
            plt.scatter(targets[i,0],targets[i,1],color='green',s=size)
        #Candidates and estimations
        plt.scatter(p_candidate[:,:,0],p_candidate[:,:,1],color='red',s=size)
        plt.scatter(best_each_y[:,0],best_each_y[:,1],color='yellow',s=size)
        plt.scatter(loc_sparse[0],loc_sparse[1],color='blue',s=size)
        
        plt.title(f'SSR result (resolution:{ssr_resol})',fontsize=f)
        plt.legend(['array','true','candidate locations','best at each y','loc_sparse'],loc="upper left",fontsize=f)
        plt.xlim([xrange[0],xrange[1]])
        plt.ylim([yrange[0],yrange[1]])
        plt.xlabel('x axis [m]',fontsize=f)
        plt.ylabel('y axis [m]',fontsize=f)
