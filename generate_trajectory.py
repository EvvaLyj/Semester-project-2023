def Calcphases(measurements,array,lmb,Phase_0):
    # n: number of antenna elements
    # Setsize: Length of trajectory
    n =array.shape[0]
    length=len(measurements)
    phases = np.zeros((length, n))
    
    if length > 1:
         for i in range(length):
            for j in range(n):
                distancetoantenna = np.linalg.norm(measurements[i]-array[j])
                phases[i, j] = (distancetoantenna * 4*math.pi/lmb + Phase_0) % (2*math.pi)
    if length == 1:
        phases = np.zeros(1,n)
        for j in range(n):
            distancetoantenna = np.linalg.norm(measurements-array[j].numpy())
            phases[0,j] = (distancetoantenna * 4*math.pi/lmb + Phase_0) % (2*math.pi)
    return phases
def GenerateTraj(Length,dt,X0,H,F,Q,R):

    real_state = []
    measurements = []
    #process(motion) model 
    #observation model
    x = X0
    for i in range(Length):
        real_state.append(x)
        x = np.dot(F,x)+np.random.multivariate_normal(mean=(0,0,0,0),cov=Q).reshape(4,1)
        measurements.append(np.array(x[0:2,:]+np.random.multivariate_normal(mean=(0,0),cov=R).reshape(2,1)))
    measurements = np.array(measurements).squeeze()
    real_state = np.array(real_state)
    lmb=2e-3
    #array of antennas
    N=10 # number of elements
    d=0.25*lmb # distance between 2 adjacent elements
    arr_dim=d*(N-1) # dimension of the array
    arr=array(N,arr_dim,0) # array elements location
    #
    Phases = Calcphases(measurements, arr,lmb,0)
    return real_state,measurements,Phases