import numpy as np
from scipy.linalg import expm


def trajectory(T_old,av_h,lv,tau,i):


	xi = np.stack((av_h[:,0],av_h[:,1],av_h[:,2],np.array(lv)),axis = 1)
	xi = np.vstack((xi,np.zeros((1,4))))
	
	exp_xi = expm(tau*xi)
	T_new = np.matmul(T_old , exp_xi)
	return T_new


