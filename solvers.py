from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
from itertools import izip

import time

def gauss_seidel(A,b,initial_guess,sparse=False,backward_substitution=True,tolerance=1e-2,maxiter=20,verbose=False):
    v = np.array(initial_guess)
    tolCheck = np.infty
    iteration = 0
    while tolCheck > tolerance and iteration < maxiter:
        t1 = time.time()
        
        v2 = np.zeros(len(v))
        indices = np.arange(len(v)-1, -1, -1) if backward_substitution else range(len(v)) # iterate forwards or backwards. default: backward substitution
        for i in indices:
            if sparse:
                Aslice = coo_matrix(A[i, :])
                Asum = 0
                if backward_substitution:
                    for c,d in izip(Aslice.col, Aslice.data):
                        if c > i: Asum += d * v2[c] # backward substitution
                        elif c < i: Asum += d * v[c] # backward substitution
                else:
                    for c,d in izip(Aslice.col, Aslice.data):
                        if c < i: Asum += d * v2[c] # forward substitution
                        elif c > i: Asum += d * v[c] # forward substitution
                v2[i] = (1.0/A[i,i]) * (b[i] - Asum)
            else:
                Aslice = A[i, :]
                if backward_substitution:
                    v2[i] = (1.0/A[i,i]) * (b[i] - np.sum(Aslice[(i+1):]*v2[(i+1):]) - np.sum(Aslice[:i]*v[:i])) # backward substitution
                else:
                    v2[i] = (1.0/A[i,i]) * (b[i] - np.sum(Aslice[:i]*v2[:i]) - np.sum(Aslice[(i+1):]*v[(i+1):])) # forward substitution
            
        #tolCheck = np.max(np.abs(v2-v)) / np.max(np.abs(v))
        tolCheck = np.sum(np.abs(v2-v)) / np.sum(np.abs(v))
        v = v2
        iteration += 1
        if verbose: 
            print "it ",iteration,"tol:",tolCheck," - t:",time.time()-t1
    return v


if __name__ == "__main__":
    A = np.array([[16,3],[7,-11]])
    b = np.array([11,13])
    x = np.array([ 0.81218274, -0.66497462])
    
    t1=time.time()
    print "linalg: t=",time.time()-t1
    xs = np.linalg.solve(A, b)
    print xs
    print np.abs(x-xs)
    
    t1=time.time()
    print "gauss-seidel: t=",time.time()-t1
    xs = gauss_seidel(A, b, [0.2, -0.1])
    print xs
    print np.abs(x-xs)