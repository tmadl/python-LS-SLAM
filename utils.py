import numpy as np

def t2v(A):
    # T2V homogeneous transformation to vector
    v = np.zeros((3,1), dtype=np.float64)
    v[:2, 0] = A[:2,2]
    v[2] = np.arctan2(A[1,0], A[0,0])
    return v

def v2t(v):
    # V2T vector to homogeneous transformation
    c = np.cos(v[2])
    s = np.sin(v[2])
    A = np.array([[c, -s, v[0]],
         [s,  c, v[1]],
         [0,  0,  1]], dtype=np.float64)
    return A

# see g2o.pdf in https://github.com/RainerKuemmerle/g2o/tree/master/doc
def apply_motion_operator(pose, motion):
    return np.array([(pose[0]+motion[0]*np.cos(motion[2])-motion[1]*np.sin(motion[2])), \
            (pose[1]+motion[0]*np.sin(motion[2])+motion[1]*np.cos(motion[2])), \
            np.mod(pose[2]+motion[2], 2*np.pi)-np.pi])
    
def get_motion_operator(pose1, pose2):
    return np.array([((pose1[0]-pose2[0])*np.cos(pose2[2])+(pose1[1]-pose2[1])*np.sin(pose2[2])),\
            (-(pose1[0]-pose2[0])*np.sin(pose2[2])+(pose1[1]-pose2[1])*np.cos(pose2[2])),\
            np.mod(pose2[2]-pose1[2], 2*np.pi)-np.pi])