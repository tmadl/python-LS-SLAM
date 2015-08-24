from utils import t2v, v2t
from solvers import *

import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import inv
import time

class PoseEdge(object):
    def __init__(self, id_from = None, id_to = None, mean = None, infm = None):
        self.id_from = id_from # viewing frame of this edge
        self.id_to = id_to # pose being observed from the viewing frame
        self.mean = mean.flatten() if type(mean) == np.ndarray else mean # Predicted virtual measurement
        self.infm = infm # Information matrix of this edge

class PoseGraph(object):
    #POSEGRAPH A class for doing pose graph optimization
    
    def __init__(self):
        # Constructor of PoseGraph
        self.nodes = [] # Pose nodes in graph. Each row has 3 values: x,y,yaw
        self.edges = [] # Edges in graph
        self.H = [] # Information matrix
        self.b = [] # Information vector
        # if x contains correct poses, then H*x = b
    
    def readGraph(self, vfile, efile):
        # Reads graph from vertex and edge file
        # vertex file
        vertices = np.loadtxt(vfile, usecols=range(1,5))
        for i in range(vertices.shape[0]):
            self.nodes.append(vertices[i, 1:4])
        self.nodes = np.array(self.nodes, dtype=np.float64)
        
        # edge file
        edges = np.loadtxt(efile, usecols=range(1,12))
        for i in range(edges.shape[0]):
            mean = edges[i, 2:5]
            infm = np.zeros((3,3), dtype=np.float64)
            # edges[i, 5:11] ... upper-triangular block of the information matrix (inverse cov.matrix) in row-major order
            infm[0,0] = edges[i, 5]
            infm[1,0] = infm[0,1] = edges[i, 6]
            infm[1,1] = edges[i, 7]
            infm[2,2] = edges[i, 8]
            infm[0,2] = infm[2,0] = edges[i, 9]
            infm[1,2] = infm[2,1] = edges[i, 10]
            edge = PoseEdge(int(edges[i,0]), int(edges[i,1]), mean, infm)
            self.edges.append(edge)
    
    def plot(self, plt=None, title=''):
        if plt is not None:
            plt.clf()
            plt.scatter(self.nodes[:, 0], self.nodes[:, 1])
            plt.title(title)
            time.sleep(0.01)
            plt.draw()
    
    def optimize(self, n_iter=1, plt=None):
        # Pose graph optimization
        
        for i_iter in range(n_iter):
            print('Pose Graph Optimization, Iteration %d.\n' % i_iter)
            
            # Create new H and b matrices each time
            self.H = np.zeros((len(self.nodes)*3,len(self.nodes)*3), dtype=np.float64)   # 3n x 3n square matrix
            self.b = np.zeros((len(self.nodes)*3,1), dtype=np.float64) # 3n x 1  column vector
            
            print('Linearizing.\n')
            self.linearize()
            
            print ('Solving.\n')
            self.solve(i_iter)
            
            if plt is not None:
                self.plot(plt, str(i_iter))

    def linearize(self):
        # Linearize error functions and formulate a linear system
        for i_edge in range(len(self.edges)):
            ei = self.edges[i_edge]
            # Get edge information
            i_node = ei.id_from
            j_node = ei.id_to
            T_z = v2t(ei.mean)
            omega = ei.infm
            
            # Get node information
            v_i = self.nodes[i_node]
            v_j = self.nodes[j_node]
            
            T_i = v2t(v_i)
            T_j = v2t(v_j)
            R_i = T_i[:2,:2]
            R_z = T_z[:2,:2]
            
            si = np.sin(v_i[2])
            ci = np.cos(v_i[2])
            dR_i = np.array([[-si, ci], [-ci, -si]], dtype=np.float64).T
            dt_ij = np.array([v_j[:2] - v_i[:2]], dtype=np.float64).T
            
            # Caluclate jacobians
            A = np.vstack((np.hstack((np.dot(-R_z.T,R_i.T), np.dot(np.dot(R_z.T, dR_i.T), dt_ij))), [0, 0, -1]))
            B = np.vstack((np.hstack((np.dot(R_z.T,R_i.T), np.zeros((2,1), dtype=np.float64))), [0, 0, 1]))
            
            # Calculate error vector
            e = t2v(np.dot(np.dot(inv(T_z), inv(T_i)), T_j))
            
            # Formulate blocks
            H_ii =  np.dot(np.dot(A.T , omega), A)
            H_ij =  np.dot(np.dot(A.T , omega), B)
            H_jj =  np.dot(np.dot(B.T , omega), B)
            b_i  = np.dot(np.dot(-A.T , omega), e)
            b_j  = np.dot(np.dot(-B.T , omega), e)
            
            # Update H and b matrix
            # #(3*(id)):(3*(id+1)) converts id to indices in H and b
            self.H[(3*i_node):(3*(i_node+1)),(3*i_node):(3*(i_node+1))] += H_ii
            self.H[(3*i_node):(3*(i_node+1)),(3*j_node):(3*(j_node+1))] += H_ij
            self.H[(3*j_node):(3*(j_node+1)),(3*i_node):(3*(i_node+1))] += H_ij.T
            self.H[(3*j_node):(3*(j_node+1)),(3*j_node):(3*(j_node+1))] += H_jj
            self.b[(3*i_node):(3*(i_node+1))] += b_i
            self.b[(3*j_node):(3*(j_node+1))] += b_j

    def solve(self, i_iter=0):
        # Solves the linear system and update all pose nodes
        print('Poses: %d, Edges: %d\n', len(self.nodes), len(self.edges))
        # The system (H b) is obtained only from relative constraints.
        # H is not full rank.
        # We solve this by anchoring the position of the 1st vertex
        # This can be expressed by adding teh equation
        # dx(1:3,1) = 0
        # which is equivalent to the following
        self.H[:3,:3] += np.eye(3)
        
        H_sparse = csr_matrix(self.H) # coo_matrix
        
        dx = spsolve(H_sparse, self.b)
        #dx = gauss_seidel(H_sparse, self.b, np.random.random(len(dx)), sparse=True)
        
        dx[:3] = [0,0,0]
        dpose = np.reshape(dx, (len(self.nodes), 3))
        
        self.nodes += dpose # update