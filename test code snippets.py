import numpy as np
from pr3_utils import *
from prediction_EKF import oneDhatmap

# Load the measurements
filename = "./data/03.npz"
# t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)    ## b = 0.6, features = 5105



A = np.zeros((4000,100))
B = np.zeros((4000,6))

C = np.hstack((A,B))
print(C.shape)