import numpy as np

V = np.diag(np.full(4,(0.1,0.1,0.1,0)))  ## V = (4,4)

def update(T,features,K,b,imu_T_cam,i):
    fsu = K[0,0]			## 552.554261
    fsv = K[1,1]
    cu = K[0,2]
    cv = K[1,2]
    ul = features[0,:,i]
    ur = features[2,:,i]
    vl = features[1,:,i]
    vr = features[3,:,i]

    v = np.reshape(np.random.normal(0,np.diag(V)),(4,1))							## v = (4,)

    z = (ul-ur)/(fsu*b)
    x = (1/fsu)*(ul - cu)*z
    y =  (1/fsv)*(vl - cv)*z

    xyz1 = np.vstack((x,y,z,np.ones((5105,))))
    world_frame = np.matmul(np.matmul(T,np.linalg.inv(imu_T_cam)),(xyz1-v))
    if(i == 0):
        print(xyz1[:,i])
    return world_frame[0],world_frame[1]

