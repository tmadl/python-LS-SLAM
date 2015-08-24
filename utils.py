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
