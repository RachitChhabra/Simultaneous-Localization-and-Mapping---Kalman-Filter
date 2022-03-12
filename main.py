import numpy as np
from scipy.linalg import expm
from pr3_utils import *
from prediction_EKF import trajectory
from update_EKF import update, skip
from tqdm import tqdm


program = 0


if __name__ == '__main__':

	# Load the measurements
	filename = "./data/03.npz"
	t,features_full,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)    ## b = 0.6, features = 5105

	features = features_full[:,0::skip,:]

	timesteps  = t[:,1:-1] - t[:,0:-2]
	
	covariance = np.zeros((3*features.shape[1] + 6, 3*features.shape[1] + 6))
	mu_pred = np.eye(4)


	T = np.zeros((4,4,timesteps.shape[1]+1))		## t shape + 1 as first T is Identity
	T[:,:,0] = np.eye(4)

	del_mu = np.zeros((6,1))

	av_hatmap = hatmap(np.transpose(angular_velocity))
	lv_hatmap = hatmap(np.transpose(linear_velocity))


	# for i in tqdm(range(0,1)):
	for i in tqdm(range(0,timesteps.shape[1])):
		## Prediction EKF
		T[:,:,i+1], mu_pred, covariance[0:6,0:6], del_mu = trajectory(T[:,:,i],mu_pred, del_mu, covariance, av_hatmap[i],lv_hatmap[i],linear_velocity[:,i],timesteps[0,i])
		
		## Update EKF
		x,y = update((T[:,:,i+1]),features,mu_pred,covariance,K,b,imu_T_cam,i)
	

	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(T,x,y, show_ori = True)

	



