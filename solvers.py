from scipy.sparse.linalg import spsolve
import numpy as np

def gauss_seidel(A,b,initial_guess,sparse=False,tolerance=1e-8,maxiter=1000):
    v = np.array(initial_guess)
    tolCheck = np.infty
    iter = 0
    while tolCheck > tolerance and iter < maxiter:
        v2 = np.zeros(len(v))
        for i in range(len(v)):
            Aslice = A[i, :]
            if sparse:
                Asum = 0
                for c,d in zip(Aslice.col, Aslice.data):
                    if c < i: Asum += d * v2[c]
                    elif c > i: Asum += d * v[c]
                v2[i] = (1.0/A[i,i]) * (b[i] - Asum)
            else:
                v2[i] = (1.0/A[i,i]) * (b[i] - np.sum(Aslice[:i]*v2[:i]) - np.sum(Aslice[(i+1):]*v[(i+1):]))
            
        tolCheck = np.max(np.abs(v2-v)) / np.max(np.abs(v))
        v = v2
        iter += 1
        if np.mod(iter, 10) == 0: print "it ",iter
    return v


if __name__ == "__main__":
    import time
    
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