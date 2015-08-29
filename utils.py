import numpy as np
import scipy.spatial


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
def apply_motion_vector(pose, motion):
    return np.array([(pose[0]+motion[0]*np.cos(motion[2])-motion[1]*np.sin(motion[2])), \
            (pose[1]+motion[0]*np.sin(motion[2])+motion[1]*np.cos(motion[2])), \
            np.mod(pose[2]+motion[2], 2*np.pi)-np.pi])
    
def get_motion_vector(pose1, pose2):
    return np.array([((pose1[0]-pose2[0])*np.cos(pose2[2])+(pose1[1]-pose2[1])*np.sin(pose2[2])),\
            (-(pose1[0]-pose2[0])*np.sin(pose2[2])+(pose1[1]-pose2[1])*np.cos(pose2[2])),\
            np.mod(pose2[2]-pose1[2], 2*np.pi)-np.pi])
    
###

def getloops(points, loopmaxdist = 1, loopmininterval = 20):
    p = np.copy(np.round(points,2)).tolist()
    D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(p))
    D[range(len(p)), range(len(p))] = np.infty
    pindex = []
    lastloop = -np.infty
    
    loopedges = []
    for i in range(len(p)):
        D2 = np.copy(D)
        D2[(i-20):(i+20), (i-20):(i+20)] = np.infty # exclude self
        closesti = np.argmin(D2[i, :])
        condition = D2[i, closesti] < loopmaxdist and closesti < i and (i-lastloop)>loopmininterval # exclude self
        #condition = D2[i, closesti] < loopmaxdist
        if condition:
            pindex.append(closesti)
            lastloop = i
            loopedges.append([i, closesti])
        else:
            pindex.append(i)
    return loopedges

ex, ey, ephi = 1e-1,1e-1,1e-1 # error standard deviations
minex, miney, minephi = 1e-6,1e-6,1e-6 # min.error standard deviations (otherwise can't invert Sigma)
def get_uncertainties_and_path(points,linear_uncertainty=ex,angular_uncertainty=ephi,ex=ex,ey=ey,ephi=ephi,minex=minex,miney=miney,minephi=minephi):
    prevtheta = 0
    uncertainties = [0]*(len(points)-1)
    path = [list(points[0])+[0]]
        
    for i in range(1, len(points)):
        dx = points[i][0]-points[i-1][0]
        dy = points[i][1]-points[i-1][1]
        try:
            if points[i][0] == points[i-1][0] and points[i][1] == points[i-1][1]: theta = 0
            else: theta = np.arctan2(dy, dx)
        except Exception,e:
            print e
            theta = 0
        dtheta = np.mod(theta - prevtheta + np.pi, np.pi*2) - np.pi
         
        sx = abs(dx)*ex+minex
        sy = abs(dy)*ey+miney
        st = abs(dtheta)*ephi+minephi
        csigma = [
        [sx**2,0,0],
        [0,sy**2,0],
        [0,0,st**2]
        ]
        uncertainties[i-1] = csigma
        prevtheta = theta
         
        v = np.sqrt(dx**2+dy**2)
        v *= 1+(linear_uncertainty*np.random.random()-linear_uncertainty/2)
        dtheta *= 1+(angular_uncertainty*np.random.random()-angular_uncertainty/2)/2
         
        ctheta = path[-1][2] + dtheta
        path.append([path[-1][0]+v*np.cos(ctheta), path[-1][1]+v*np.sin(ctheta), ctheta])
        
    sx = minex
    sy = miney
    st = minephi
    uncertainties.append([
    [sx**2,0,0],
    [0,sy**2,0],
    [0,0,st**2]
    ])
    path = np.array(path)
    path -= path[0, :]
    return np.array(uncertainties), path

def readPoseGraph(pfile):
    # Reads graph from vertex and edge file (g2o format - see https://github.com/RainerKuemmerle/g2o)
    # vertex file
    #lines = np.genfromtxt(pfile)
    with open(pfile, 'r') as f:
        lines = f.readlines()
    odometry_poses = []
    constraints = []
    for i in range(len(lines)):
        lines[i] = lines[i].split(' ')
        typ = str(lines[i][0]).lower()
        if 'vertex' in typ:
            odometry_poses.append([float(l) for l in lines[i][2:5]])
        elif 'edge' in typ:
            line = [0] + [float(l) for l in lines[i][1:]]
            
            mean = line[2:5]
            infm = np.zeros((3,3), dtype=np.float64)
            # edges[i, 5:11] ... upper-triangular block of the information matrix (inverse cov.matrix) in row-major order
            infm[0,0] = line[5]
            infm[1,0] = infm[0,1] = line[6]
            infm[1,1] = line[7]
            infm[2,2] = line[8]
            infm[0,2] = infm[2,0] = line[9]
            infm[1,2] = infm[2,1] = line[10]
            constraints.append([int(line[0]), int(line[1]), mean, infm])
    odometry_poses = np.array(odometry_poses)
    return odometry_poses, constraints

def writePoseGraph(odometry, constraints, pfile):
    with open(pfile, 'w') as f:
        vi = 0
        for o in odometry:
            f.write('VERTEX2 '+str(vi)+' '+str(o[0])+' '+str(o[1])+' '+str(o[2])+'\n')
            vi += 1
        for e in constraints:
            s = 'EDGE2 '+str(e[0])+' '+str(e[1])+' ' # first FROM then TO
            for m in e[2]: s+=str(m)+' '
            infm = e[3]
            s+=str(infm[0,0])+' '
            s+=str(infm[1,0])+' '
            s+=str(infm[1,1])+' '
            s+=str(infm[2,2])+' '
            s+=str(infm[0,2])+' '
            s+=str(infm[1,2])+'\n'
            f.write(s)