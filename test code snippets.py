import numpy as np
from pr3_utils import *

# Load the measurements
filename = "./data/03.npz"
t,features_full,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)    ## b = 0.6, features = 5105

Ks = np.zeros((4,4))
Ks[0:3,0:3] = K
Ks[3,1:3] = K[1,1:3]
Ks[2,3] = -K[0,0]*b

print(K)
print(Ks)


