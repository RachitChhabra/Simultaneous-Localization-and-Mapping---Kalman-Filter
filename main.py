import numpy as np
from scipy.linalg import expm
from pr3_utils import *
from prediction_EKF import trajectory



if __name__ == '__main__':

	# Load the measurements
	filename = "./data/03.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)    ## b = 0.6, features = 5105

	timesteps  = t[:,1:-1] - t[:,0:-2]
	
	covariance = np.zeros((2*features.shape[1] + 6, 2*features.shape[1] + 6))
	mu = np.zeros((3*features.shape[1],3*features.shape[1]))


	# fsu = K[0,0]			## 552.554261
	# ul = features[0,0,0]
	# ur = features[2,0,0]
	# z = (fsu*b)/(ul-ur)

	T = np.zeros((4,4,timesteps.shape[1]+1))		## t shape + 1 as first T is Identity
	T[:,:,0] = np.eye(4)

	av_hatmap = hatmap(np.transpose(angular_velocity))

	for i in range(0,timesteps.shape[1]):
		T[:,:,i+1] = trajectory(T[:,:,i],av_hatmap[i],linear_velocity[:,i],timesteps[0,i],i)
		

	# You can use the function below to visualize the robot pose over time
	visualize_trajectory_2d(T, show_ori = True)





