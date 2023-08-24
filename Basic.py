"""
Basic.py

Here are some basic functions used in this project.
"""

import numpy as np
import math

def array(num_elements,dim,x_start):
    """
    Generate the positions of a linear array lying on the x-axis.
    num_elements: number of antenna elements.
    dim: the total dimension of this linear array
    x_start: the x-position of the first antenna element. 
    """
    lin_array=np.zeros((num_elements,2))
    lin_array[:,0]=[i for i in np.linspace(x_start,x_start+dim,num_elements)]#y_axis
    lin_array[:,1]=[ 0 for _ in range(num_elements)]#x_axis
    return lin_array


def near_field_Boundary(lamb,arr_dim):
    """
    The boundary between the Fresnel near-field region and far-field region.
    """
    return 2*arr_dim**2/lamb

def sample_in_half_circle(R,N):
    """
    Random sampling N times within a half circle of radius R. 
    """
    samples=np.zeros((N,2))
    count=0
    while(count<N):       
        x=(np.random.uniform(0,1,1)-0.5)*2*R#U[-R,R]
        y=np.random.uniform(0,1,1)*R# U[0,R]
        if(x**2+y**2<=R**2):
            samples[count]=np.array([x,y]).reshape(2,)
            count+=1
    return samples

def Noise_dB_to_std(noise_dB):
    """
    Converting a series of noise in dB into the standard deviation. 
    """
    noise_std=np.empty(len(noise_dB))
    for i in range(len(noise_dB)):
        noise_std[i]=math.pow(10,-noise_dB[i]/20)
    return noise_std

def Single_Noise_dB_to_std(noise_dB):
    """
    Converting one noise in dB into the standard deviation. 
    """
    noise_std=math.pow(10,-noise_dB/20)
    return noise_std