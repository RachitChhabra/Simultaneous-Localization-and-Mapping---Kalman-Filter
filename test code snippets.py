import numpy as np
from pr3_utils import *

# Load the measurements
filename = "./data/03.npz"
# t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)    ## b = 0.6, features = 5105

A = np.zeros((4,8))
A[1] = 1
print(A)
B = np.reshape(np.transpose(A),(32,1))
print(B)