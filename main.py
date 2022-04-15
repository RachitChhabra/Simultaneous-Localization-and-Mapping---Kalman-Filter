import numpy as np
from pr3_utils import *
from prediction_EKF import trajectory
from update_EKF import update, skip
from tqdm import tqdm

program = 3

'''
Change the value of program to run different parts of the code.
(a) For IMU Localization via EKF Prediction
		program == 1
(b) For IMU Mapping via EKF Update
		program == 2
(c) For VI SLAM
		program == 3
'''

if __name__ == '__main__':

	# Load the measurements
	filename = "./data/10.npz"
	t,features_full,linear_velocity,angular_velocity,K,b,imu_T_cam1 = load_data(filename)    ## b = 0.6, features = 5105

	Rot_about_x = np.zeros((4,4))
	Rot_about_x[2,2] = Rot_about_x[1,1] = -1
	Rot_about_x[0,0] = Rot_about_x[3,3] = 1
	imu_T_cam = Rot_about_x@imu_T_cam1

	features = features_full[:,0::skip,:]
	f = features.shape[1]
	timesteps  = t[:,1:-1] - t[:,0:-2]
	
	covariance = np.zeros((3*features.shape[1] + 6, 3*features.shape[1] + 6))
	covariance[-6:,-6:] = np.eye(6)*0.001

	mu_pred = np.eye(4)

	T = np.zeros((4,4,timesteps.shape[1]+1))		## t shape + 1 as first T is Identity
	T[:,:,0] = np.eye(4)

	del_mu = np.zeros((6,1))

	av_hatmap = hatmap(np.transpose(angular_velocity))
	lv_hatmap = hatmap(np.transpose(linear_velocity))
	x = 0
	y = 0

	if ((program == 1)|(program == 2)|(program == 3) ):
		for i in tqdm(range(0,timesteps.shape[1])):
			## Prediction EKF
			T[:,:,i+1], mu_pred, covariance, del_mu = trajectory(T[:,:,i],mu_pred, del_mu, covariance, av_hatmap[i],lv_hatmap[i],linear_velocity[:,i],timesteps[0,i],program)

			if (program > 1):
				## Update EKF
				x,y,mu_pred,covariance = update((T[:,:,i+1]),features,mu_pred,covariance,K,b,imu_T_cam,i,program)

		# You can use the function below to visualize the robot pose over time
		visualize_trajectory_2d(T,x,y, show_ori = True)
	else:
		print("Please choose the correct program")

