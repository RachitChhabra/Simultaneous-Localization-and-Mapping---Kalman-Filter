import numpy as np
from pr3_utils import *
from prediction_EKF import trajectory
from update_EKF import update, skip
from tqdm import tqdm
from scipy.linalg import expm


if __name__ == '__main__':
    # Load the measurements
    filename = "./data/03.npz"
    t,features_full,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)    ## b = 0.6, features = 5105
    features = features_full[:,0::skip,:]
    f = features.shape[1]
    timesteps  = t[:,1:-1] - t[:,0:-2]
    # covariance = np.zeros((3*features.shape[1] + 6, 3*features.shape[1] + 6))
    covariance = np.eye(3*features.shape[1] + 6)*0.0001
    covariance[-6:,-6:] = 0
    mu_pred = np.eye(4)
    T = np.zeros((4,4,timesteps.shape[1]+1))		## t shape + 1 as first T is Identity
    T[:,:,0] = np.eye(4)
    del_mu = np.zeros((6,1))
    av_hatmap = hatmap(np.transpose(angular_velocity))
    lv_hatmap = hatmap(np.transpose(linear_velocity))
    
    for i in tqdm(range(0,timesteps.shape[1])):
        
        # mu_pred = mu_pred*expm(timesteps[0,i]*np.vstack((np.stack((av_hatmap[i,:,0],av_hatmap[i,:,1],av_hatmap[i,:,2],np.array(linear_velocity[:,i])),axis = 1),np.zeros((1,4)))))
        # T[:,:,i+1] = np.matmul(mu_pred,T[:,:,i])
        T[:,:,i+1], mu_pred, covariance, del_mu = trajectory(T[:,:,i],mu_pred, del_mu, covariance, av_hatmap[i],lv_hatmap[i],linear_velocity[:,i],f,timesteps[0,i])
        
	# You can use the function below to visualize the robot pose over time
    visualize_trajectory_2d(T, show_ori = True)

def trajectory(T_old,mu_pred_old,del_old,covariance,av_h,lv_h,lv,f,tau):
	## We only need top left 6x6 covariance matrix

    mu_pred_new = mu_pred_old*expm(tau*np.vstack((np.stack((av_h[:,0],av_h[:,1],av_h[:,2],np.array(lv)),axis = 1),np.zeros((1,4)))))
    T_new = np.matmul(mu_pred_new,T_old)
    
    return T_new, mu_pred_new

def oneDhatmap(vector):
    '''
    Input:
        1 x 3 vector
    Output:
        converts it to a 3 x 3 hat map and outputs 3x3
    '''
    hatmap = np.zeros((3,3))
  
    hatmap[2,1] = vector[0,0]                 ##  First element
    hatmap[0,2] = vector[0,1]                 ## Second element
    hatmap[1,0] = vector[0,2]                 ##  Third element
    
    hatmap[1,2] = -hatmap[2,1]
    hatmap[2,0] = -hatmap[0,2]
    hatmap[0,1] = -hatmap[1,0]
    return hatmap

def four_four_hatmap(vector):
	'''
	Input:
		6 x 1 vector
	Output:
		converts it to a 4 x 4 hat map and outputs 4 x 4
	'''
	w_hat = oneDhatmap(np.transpose(vector[0:3]))

	four_four = np.hstack((w_hat,vector[3:6]))
	four_four = np.vstack((four_four,np.zeros((1,4))))
	return four_four
