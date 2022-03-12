import numpy as np
from prediction_EKF import oneDhatmap



from pr3_utils import *
from prediction_EKF import trajectory
from tqdm import tqdm



V = np.diag(np.full(4,(100,100,100,0)))  ## V = (4,4)

skip = 20
file = '03'

if(file == '03'):
    m = int(5105/skip)+1
if(file == '10'):
    m = int(13289/skip)+1

m_sum = np.zeros((4,m))
world_frame_mean = np.zeros((4,m)) 
world_frame_mean[:,:] = np.nan
m_ticker =   np.zeros((1,m))

Pt = np.vstack((np.eye(3),np.zeros((1,3))))

def update(T,features,covariance,K,b,imu_T_cam,i):
    fsu = K[0,0]			## 552.554261
    fsv = K[1,1]
    cu = K[0,2]
    cv = K[1,2]
    ul = features[0,:,i]            ## ul -> (5015,)
    ur = features[2,:,i]
    vl = features[1,:,i]
    vr = features[3,:,i]

    ## Construct Ks
    Ks = np.zeros((4,4))
    Ks[0:3,0:3] = K
    Ks[3,1:3] = K[1,1:3]
    Ks[2,3] = -K[0,0]*b

    ## Noise to xyz1
    v = np.reshape(np.random.normal(0,np.diag(V)),(4,1))							## v = (4,1)

    ## Optical Frame coordinates
    z = (fsu*b)/(ul-ur)
    x = (1/fsu)*(ul - cu)*z
    y =  (1/fsv)*(vl - cv)*z
    xyz1 = np.vstack((x,y,z,np.ones(x.shape)))

    ## Calculating mean in world frame
    world_frame = np.matmul(np.matmul(T,imu_T_cam),(xyz1-v))
    # world_frame = np.matmul(np.matmul(T,imu_T_cam),(xyz1))

    nnan_index = np.where(ul!= -1)
    m_sum[:,nnan_index] += world_frame[:,nnan_index]

    # if(i == 0):
    world_frame_mean[:,nnan_index] = m_sum[:,nnan_index]/(m_ticker[:,nnan_index] + 1)
    
    m_ticker[:,nnan_index] += 1
    Nt_index = np.array(nnan_index).flatten()
    Nt = Nt_index.shape[0]


    ## Reshaping from 4xm to to 3mx1
    mean_3m_1 = np.transpose(world_frame_mean[0:3,:]).reshape((3*m,1))        ## mean_3m_1 -> (3m,1)
    mean_4m_1 = np.transpose(world_frame_mean[0:4,:]).reshape((4*m,1))        ## mean_4m_1 -> (4m,1)

    z_tilda = np.zeros((4*Nt,1))
    l = 0
    H = np.zeros((4*Nt,3*m))
    H_imu = np.zeros((4*Nt,6))

    for j in Nt_index:
        z_tilda[4*l:4*(l+1)] = np.matmul(Ks,np.matmul(np.linalg.inv(imu_T_cam),np.matmul(np.linalg.inv(imu_T_cam),mean_4m_1[4*j:4*(j+1)])))/mean_4m_1[4*j+2]
        
        
        dpdp = dpibydq(np.matmul(np.linalg.inv(imu_T_cam),np.matmul(np.linalg.inv(T),mean_4m_1[4*j:4*(j+1)])))
        H[4*l:4*(l+1),3*j:3*(j+1)] = np.matmul(Ks,np.matmul(dpdp,np.matmul(np.linalg.inv(imu_T_cam),np.matmul(np.linalg.inv(T),Pt))))
        
        H_imu[4*l:4*(l+1),:] = - np.matmul(Ks,np.matmul(dpdp,np.matmul(np.linalg.inv(imu_T_cam),specialplus(np.transpose(np.matmul(np.linalg.inv(T),mean_4m_1[4*j:4*(j+1)]))))))
        



        l+=1
    
 
    ## Calculating Kalman Gain
    sigma_Ht = np.matmul(covariance[6:3*m+6,6:3*m+6],np.transpose(H))
    I_star_V = np.eye(4*Nt)

    K = np.matmul(sigma_Ht,np.linalg.inv(np.matmul(H,sigma_Ht)+I_star_V))

    mean_3m_1 = mean_3m_1 + np.matmul(K,features[:,Nt_index,i].reshape(z_tilda.shape) - z_tilda)
    covariance[6:3*m+6,6:3*m+6] = np.matmul((np.eye(3*m) - np.matmul(K,H)),covariance[6:3*m+6,6:3*m+6]) 
   
    world_frame_mean[0:3,:] = np.reshape(mean_3m_1,(3,m),order = 'F')
    

    return world_frame_mean[0],world_frame_mean[1]



def specialplus(s):
    sp = np.hstack((np.eye(3), -oneDhatmap(s)))
    sp = np.vstack((sp,np.zeros((1,6))))
    return sp

def dpibydq(q):
    dq = np.zeros((4,4))
    dq[0,0] = 1/q[2]
    dq[0,2] = -q[0] / (q[2] * q[2])
    dq[1,1] = 1/q[2]
    dq[1,2] = -q[1] / (q[2] * q[2])
    dq[3,2] = -q[3] / (q[2] * q[2])
    dq[3,3] = 1/q[2]
    return dq





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
		x,y = update((T[:,:,i+1]),features,covariance,K,b,imu_T_cam,i)
	

	# You can use the function below to visualize the robot pose over time
	visualize_trajectory_2d(T,x,y)#, show_ori = True)

