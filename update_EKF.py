import numpy as np
from prediction_EKF import oneDhatmap,four_four_hatmap
from scipy.linalg import expm

V = np.diag(np.full(4,(0.01,0.01,0.01,0.01)))  ## V = (4,4)

skip = 16
file = '10'

if(file == '03'):
    m = int(5105/skip)+1
if(file == '10'):
    m = int(13289/skip)+1

world_frame_mean = np.zeros((4,m)) 
world_frame_mean[:,:] = 0
m_ticker =   np.zeros((1,m))

Pt = np.vstack((np.eye(3),np.zeros((1,3))))

def update(T,features,mu_imu,covariance,K,b,imu_T_cam,i,program):

    ## Noise to features
    v = np.reshape(np.random.normal(0,np.diag(V)),(4,1))							## v = (4,1)
    nnan_index = np.where(features[1,:,i]!= -1)

    features[:,nnan_index[0],i] -= v

    Nt_index = np.array(nnan_index).flatten()
    Nt = Nt_index.shape[0]

    # print(features[:,nnan_index,i])
    fsu = K[0,0]			        ## 552.554261
    fsv = K[1,1]
    cu = K[0,2]
    cv = K[1,2]
    ul = features[0,:,i]            ## ul -> (5015,)
    ur = features[2,:,i]
    vl = features[1,:,i]
    vr = features[3,:,i]

    ## Construct Ks
    Ks = np.zeros((4,4))
    Ks[0,0] = K[0,0]
    Ks[0,2] = K[0,2]
    Ks[1,1] = K[1,1]
    Ks[1,2] = K[1,2]
    Ks[2,0] = K[0,0]
    Ks[2,2] = K[0,2]
    Ks[2,3] = -1 * b * K[0,0]
    Ks[3,1] = K[1,1]
    Ks[3,2] = K[1,2]

    ## Optical Frame coordinates
    z = (fsu*b)/(ul-ur)
    x = (1/fsu)*(ul - cu)*z
    y =  (1/fsv)*(vl - cv)*z
    xyz1 = np.vstack((x,y,z,np.ones(x.shape)))

    ## Calculating mean in world frame
    world_frame = np.matmul(np.matmul(T,imu_T_cam),(xyz1))
 
    ## Finding new features
    not_seen_yet = np.where(m_ticker[0,:] == 0)
    first_time_seen = np.intersect1d(nnan_index,not_seen_yet)
    world_frame_mean[:,first_time_seen] = world_frame[:,first_time_seen]
    m_ticker[:,first_time_seen] += 1

    ## Reshaping from 4xm to to 3mx1
    mean_3m_1 = np.transpose(world_frame_mean[0:3,:]).reshape((3*m,1))        ## mean_3m_1 -> (3m,1)
    mean_4m_1 = np.transpose(world_frame_mean[0:4,:]).reshape((4*m,1))        ## mean_4m_1 -> (4m,1)
    # print(mean_3m_1[0:24])
    z_tilda = np.zeros((4*Nt,1))
    l = 0
    H_stereo = np.zeros((4*Nt,3*m))
    H_imu = np.zeros((4*Nt,6))

    for j in Nt_index:
        q = np.matmul(np.linalg.inv(imu_T_cam),np.matmul(np.linalg.inv(mu_imu),mean_4m_1[4*j:4*(j+1)]))
        z_tilda[4*l:4*(l+1)] = np.matmul(Ks,q/q[2])

        dpdp = dpibydq(np.matmul(np.linalg.inv(imu_T_cam),np.matmul(np.linalg.inv(mu_imu),mean_4m_1[4*j:4*(j+1)])))
        H_stereo[4*l:4*(l+1),3*j:3*(j+1)] = np.matmul(Ks,np.matmul(dpdp,np.matmul(np.linalg.inv(imu_T_cam),np.matmul(np.linalg.inv(mu_imu),Pt))))
        
        if(program == 3):
            H_imu[4*l:4*(l+1),:] = - np.matmul(Ks,np.matmul(dpdp,np.matmul(np.linalg.inv(imu_T_cam),specialplus(np.transpose(np.matmul(np.linalg.inv(mu_imu),mean_4m_1[4*j:4*(j+1)]))))))
            
        if(covariance[3*j,3*j] == 0):
            covariance[3*j:3*(j+1),3*j:3*(j+1)] = np.eye(3)*0.001
        l+=1

    if (program == 2):
    # Calculating Kalman Gain

        sigma_Ht_stereo = np.matmul(covariance[6:3*m+6,6:3*m+6],np.transpose(H_stereo))
        I_star_V = np.eye(4*Nt)
        K_stereo = np.matmul(sigma_Ht_stereo,np.linalg.inv(np.matmul(H_stereo,sigma_Ht_stereo)+I_star_V))
        mean_3m_1 = mean_3m_1 + np.matmul(K_stereo,features[:,Nt_index,i].reshape(z_tilda.shape) - z_tilda)
        covariance[6:3*m+6,6:3*m+6] = np.matmul((np.eye(3*m) - np.matmul(K_stereo,H_stereo)),covariance[6:3*m+6,6:3*m+6]) 
   

    if (program == 3):
        H_slam = np.hstack((H_stereo,H_imu))        # (4 Nt , 3m+6)
    
        sigma_Ht_slam = np.matmul(covariance,np.transpose(H_slam))
        I_star_V_slam = np.eye(4*Nt)*V[0,0]
        K_slam = np.matmul(sigma_Ht_slam,np.linalg.inv(np.matmul(H_slam,sigma_Ht_slam)+I_star_V_slam))   # (3m+6 , 4 Nt)

        eye_minus_KH = np.eye(3*m+6) - np.matmul(K_slam,H_slam)
        
        covariance = np.matmul(eye_minus_KH,np.matmul(covariance,np.transpose(eye_minus_KH))) + np.matmul(K_slam,np.matmul(I_star_V_slam,np.transpose(K_slam)))

        innovation = np.reshape(np.squeeze(features[:,nnan_index,i]),z_tilda.shape,order = 'F') - z_tilda

        mean_3m_1 = mean_3m_1 + np.matmul(K_slam[0:3*m,:],innovation)
        
        mu_imu   = np.matmul(mu_imu,expm(four_four_hatmap(np.matmul(K_slam[-6:,:],innovation))))

    world_frame_mean[0:3,:] = np.reshape(mean_3m_1,(3,m),order = 'F')

    return world_frame_mean[0],world_frame_mean[1],mu_imu,covariance

def specialplus(s):
    sp = np.hstack((np.eye(3), -oneDhatmap(s)))
    sp = np.vstack((sp,np.zeros((1,6))))
    return sp

def dpibydq(q):
    dq = np.zeros((4,4))
    dq[0,0] = 1/q[2]
    dq[0,2] = -q[0]/(q[2]*q[2])
    dq[1,1] = 1/q[2]
    dq[1,2] = -q[1]/(q[2]*q[2])
    dq[3,2] = -q[3]/(q[2]*q[2])
    dq[3,3] = 1/q[2]
    return dq

