import numpy as np
from pr3_utils import *
from prediction_EKF import oneDhatmap

# Load the measurements
filename = "./data/03.npz"
# t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)    ## b = 0.6, features = 5105


A = np.matrix([[1.1,0,1.732],[0,0.2,0.1],[1.732,0.1,19.1]])


print(A)