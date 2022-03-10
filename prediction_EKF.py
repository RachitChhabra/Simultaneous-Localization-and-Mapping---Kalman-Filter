import numpy as np
from scipy.linalg import expm
from pr3_utils import hatmap

W = np.diag(np.full(6,(0.3, 0.3, 0.3, 0.05, 0.05, 0.05 )))  ## W = (6,6)

def trajectory(T_old,mu_pred_old,del_old,covariance_old,av_h,lv_h,lv,tau):
	## We only need top left 6x6 covariance matrix

	
	w = np.reshape(np.random.normal(0,np.diag(W)),(6,1))							## w = (6,)


	xi = np.stack((av_h[:,0],av_h[:,1],av_h[:,2],np.array(lv)),axis = 1)
	xi = np.vstack((xi,np.zeros((1,4))))
	exp_xi = expm(tau*xi)
	mu_pred_new = mu_pred_old*exp_xi

	curly_hatmap = np.hstack((av_h, lv_h))
	curly_hatmap = np.vstack((curly_hatmap, (np.hstack((np.zeros((3,3)), av_h)))))	## curly hatmap -> 6 x 6
	exp_curly = expm(-tau*curly_hatmap)


	del_new = np.matmul(exp_curly,del_old) + w 

	mu_pred_new = np.matmul(mu_pred_old,exp_xi)
	new_covariance = np.matmul(exp_curly,np.matmul(covariance_old[0:6,0:6],np.transpose(exp_curly))) + W
	
	del_new_1_3_h = oneDhatmap(np.transpose(del_new[0:3]))

	a1 = np.transpose(np.matrix(del_new_1_3_h[:,0]))
	a2 = np.transpose(np.matrix(del_new_1_3_h[:,1]))
	a3 = np.transpose(np.matrix(del_new_1_3_h[:,2]))
	
	xi_del = np.hstack((a1,a2,a3,del_new[3:6]))
	
	xi_del = np.vstack((xi_del,np.zeros((1,4))))
	exp_xi_del = expm(xi_del)

	T_new = np.matmul(mu_pred_new,exp_xi_del)


	return T_new, mu_pred_new, new_covariance, del_new




def oneDhatmap(vector):
    '''
    Input:
        t x 3 vector
    Output:
        converts it to a t x 3 x 3 hat map and outputs 3x3xt
    '''
    hatmap = np.zeros((3,3))
  
    hatmap[2,1] = vector[0,0]                 ##  First element
    hatmap[0,2] = vector[0,1]                 ## Second element
    hatmap[1,0] = vector[0,2]                 ##  Third element
    
    hatmap[1,2] = -hatmap[2,1]
    hatmap[2,0] = -hatmap[0,2]
    hatmap[0,1] = -hatmap[1,0]
    return hatmap
