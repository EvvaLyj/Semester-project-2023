import numpy as np

def array(num_elements,dim,x_start):
    lin_array=np.zeros((num_elements,2))
    lin_array[:,0]=[i for i in np.linspace(x_start,x_start+dim,num_elements)]#y_axis
    lin_array[:,1]=[ 0 for _ in range(num_elements)]#x_axis
    return lin_array


def near_field_Boundary(lamb,arr_dim):
    return 2*arr_dim**2/lamb

def sample_in_half_circle(R,N):
    samples=np.zeros((N,2))
    count=0
    while(count<N):       
        x=(np.random.uniform(0,1,1)-0.5)*2*R#U[-R,R]
        y=np.random.uniform(0,1,1)*R# U[0,R]
        if(x**2+y**2<=R**2):
            samples[count]=np.array([x,y]).reshape(2,)
            count+=1
    return samples