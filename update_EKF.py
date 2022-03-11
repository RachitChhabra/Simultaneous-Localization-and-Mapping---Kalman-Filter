import numpy as np
from prediction_EKF import oneDhatmap

V = np.diag(np.full(4,(50,50,50,0)))  ## V = (4,4)

m_sum = np.zeros((4,5105))
world_frame_mean = np.zeros((4,5105)) 
world_frame_mean[:,:] = np.nan
m_ticker =   np.zeros((1,5105))

def update(T,features,K,b,imu_T_cam,i):
    fsu = K[0,0]			## 552.554261
    fsv = K[1,1]
    cu = K[0,2]
    cv = K[1,2]
    ul = features[0,:,i]            ## ul -> (5015,)
    ur = features[2,:,i]
    vl = features[1,:,i]
    vr = features[3,:,i]

    ## Construct Ks
    Ks = np.zeros((4,4))
    Ks[0:3,0:3] = K
    Ks[3,1:3] = K[1,1:3]
    Ks[2,3] = -K[0,0]*b

    ## Noise to xyz1
    v = np.reshape(np.random.normal(0,np.diag(V)),(4,1))							## v = (4,1)

    ## Optical Frame coordinates
    z = (fsu*b)/(ul-ur)
    x = (1/fsu)*(ul - cu)*z
    y =  (1/fsv)*(vl - cv)*z

    xyz1 = np.vstack((x,y,z,np.ones(x.shape)))

    ## Calculating mean in world frame
    world_frame = np.matmul(np.matmul(T,imu_T_cam),(xyz1-v))
    nnan_index = np.where(ul!= -1)
    m_sum[:,nnan_index] += world_frame[:,nnan_index]
    world_frame_mean[:,nnan_index] = m_sum[:,nnan_index]/(m_ticker[:,nnan_index] + 1)
    m_ticker[:,nnan_index] += 1

    ## Reshaping from 4xm to to 3mx1
    for i in range()






    return world_frame_mean[0],world_frame_mean[1]



def specialplus(s):
    sp = np.hstack((np.eye(3), -oneDhatmap(s)))
    sp = np.vstack((sp,np.zeros((1,6))))
    print(sp)