import numpy as np
from pr3_utils import *
from prediction_EKF import oneDhatmap

# Load the measurements
filename = "./data/03.npz"
# t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)    ## b = 0.6, features = 5105



s = np.zeros((3,1))
a = np.where(s[:,0] == 0)

t = np.zeros((2,1))
b = np.where(t[:,0] == 0)

c = np.intersect1d(a,b)
print(c)
