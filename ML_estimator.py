import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




class ML:
    def ML_grid(self,xrange,yrange,phase_obs,arr,lamb,resolution,flag_plot=True):
        N=arr.shape[0]

        num_grid_x=int(np.floor((xrange[1]-xrange[0])/resolution))+1
        num_grid_y=int(np.floor((yrange[1]-yrange[0])/resolution))+1

        grid_x, step_x=np.linspace(xrange[0],xrange[1],num_grid_x,retstep=True)
        grid_y, step_y=np.linspace(yrange[0],yrange[1],num_grid_y,retstep=True)

        objective_values=np.zeros((num_grid_x,num_grid_y))
        for i in range(num_grid_x):
            for j in range(num_grid_y):
                p=np.array([grid_x[i],grid_y[j]])
                distance=np.array([np.linalg.norm(arr[k]-p) for k in range(N)])
                phase_obs_diff=phase_obs-phase_obs[0]
                phase_model_diff=4*np.pi/lamb*(distance-distance[0])
                objective_values[i,j]=np.cos(phase_obs_diff-phase_model_diff).sum()

        a,b=np.where(objective_values==np.max(objective_values))
        loc=np.array([grid_x[a[0]],grid_y[b[0]]])
        
#         print(f"ML estimation is{loc}")


        if(flag_plot):
            fig,ax=plt.subplots(figsize=(8,6))
            df = pd.DataFrame( objective_values.T,
                          columns=np.round(grid_x,3),
                          index=np.round(grid_y,3)
                          )
            #plotting
            sns.set_context({"figure.figsize":(5,5)})
            sns.heatmap(data=df,square=True,cmap="gist_gray")

            #ax.axes.yaxis.set_ticks([])
            #ax.axes.xaxis.set_ticks([])
            plt.xlabel('x')
            plt.ylabel('y')

            plt.scatter(a,b,c="red")
            plt.title(f'Grid searching result with resolution= {resolution/lamb}wavelength')
        
        loc=loc.reshape([-1,2])
        return np.array([num_grid_x,num_grid_y]),loc
    
    def ML_grid2(self,xrange,yrange,phase_obs,arr,lamb,resolution,num_ite,flag_plot=True):
        N=arr.shape[0]
        #hierarchical grid searching
        reso=resolution
        search_left_bound=xrange[0]
        search_right_bound=xrange[1]
        search_lower_bound=yrange[0]
        search_upper_bound=yrange[1]
        locs=[]
        
        for k in range(num_ite):
            
            num_grid_x=int(np.floor((search_right_bound-search_left_bound)/reso))
            num_grid_y=int(np.floor((search_upper_bound-search_lower_bound)/reso))

#             print(num_grid_x,num_grid_y)
            
            
            grid_x=np.arange(search_left_bound+0.5*reso,search_left_bound+(num_grid_x)*reso,reso)
            grid_y=np.arange(search_lower_bound+0.5*reso,search_lower_bound+(num_grid_y)*reso,reso)
#             print(len(grid_x),len(grid_y))
            objective_values=np.zeros((num_grid_x,num_grid_y))
            for i in range(num_grid_x):
                for j in range(num_grid_y):
                    p=np.array([grid_x[i],grid_y[j]])
                    distance=np.array([np.linalg.norm(arr[k]-p) for k in range(N)])
                    phase_obs_diff=phase_obs-phase_obs[0]
                    phase_model_diff=4*np.pi/lamb*(distance-distance[0])
                    objective_values[i,j]=np.cos(phase_obs_diff-phase_model_diff).sum()

            a,b=np.where(objective_values==np.max(objective_values))
            loc=np.array([grid_x[a[0]],grid_y[b[0]]])
            locs=np.append(locs, loc)


            if(flag_plot):
                plt.figure()
                fig,ax=plt.subplots(figsize=(8,6))
                df = pd.DataFrame( objective_values.T,
                              columns=np.round(grid_x,3),
                              index=np.round(grid_y,3)
                              )
                #plotting
                sns.set_context({"figure.figsize":(5,5)})
                sns.heatmap(data=df,square=True,annot=False,cmap="gist_gray")

                plt.xlabel('x')
                plt.ylabel('y')

                plt.text(a+0.5,b+0.5,'target',c="red",ha='center',va='center')
                ax.add_patch(plt.Rectangle((a-2,b-2), 5,5, color="blue", fill=False, linewidth=1))
                plt.title(f'Searching result with resolution= {reso/lamb}wavelength,ite={k+1}')

#                 plt.savefig(f'C:/Users/73160/OneDrive/项目/project/Weekly reports/Sp_4/ML2_Result_{k+1}.png')
            
            #increase the resolution and search within the best grid
            
            search_left_bound=max(loc[0]-reso*2.5,search_left_bound)
            search_right_bound=min(loc[0]+reso*2.5,search_right_bound)
            search_lower_bound=max(loc[1]-reso*2.5,search_lower_bound)
            search_upper_bound=min(loc[1]+reso*2.5,search_upper_bound)
            reso=0.25*reso
#             print(reso,search_left_bound,search_right_bound,search_lower_bound,search_upper_bound)
            
        locs=locs.reshape([-1,2])
        return np.max(objective_values),locs
    
    def plotting(self,size,f,targets,Loc_MLgrid,arr,xrange,yrange):
        plt.figure(figsize=(4,4))
        plt.scatter(arr[:,0],arr[:,1],color='black')
        for i in range(len(targets)):
            plt.scatter(targets[i,0],targets[i,1],color='green',s=size)
        for j in range(Loc_MLgrid.shape[0]):
            plt.scatter(Loc_MLgrid[j,0],Loc_MLgrid[j,1],color='red',s=size)
            plt.text(Loc_MLgrid[j,0],Loc_MLgrid[j,1],j+1)
        plt.scatter(Loc_MLgrid[0],Loc_MLgrid[1],color='red',s=size)
        plt.title('ML result',fontsize=f)
        plt.legend(['array','true','loc_grid'],loc='upper left',fontsize=f)
        plt.xlim([xrange[0],xrange[1]])
        plt.ylim([yrange[0],yrange[1]])
        plt.xlabel('x axis [m]',fontsize=f)
        plt.ylabel('y axis [m]',fontsize=f)
        
    
 