import numpy as np
import matplotlib.pyplot as plt
from utils import *
from posegraph import PoseGraph, PoseEdge

cols = [(np.random.random(), np.random.random(), np.random.random()) for i in range(100)]

coords = []
loopedges = []

def runslam(coords, loopedges):
    uncertainties, path = get_uncertainties_and_path(coords, linear_uncertainty=0, angular_uncertainty=0)
    
    pg = PoseGraph() # LS-SLAM pose graph
    edges = []
    # populate with data -  path
    for p in range(len(path)):
        pg.nodes.append(path[p])
        if p > 0:
            covm = uncertainties[p]#+uncertainties[p-1]
            infm=np.linalg.inv(covm)
            edge = [p-1, p, get_motion_vector(path[p], path[p-1]), infm]
            edges.append(edge)
            pg.edges.append(PoseEdge(*edge))
    pg.nodes = np.array(pg.nodes)
    # populate with data - loop closures
    for idpair in loopedges:   
        cov_prec = [
        [minex/1e10,0,0], # very sure about x and y
        [0,miney/1e10,0],
        [0,0,ephi] 
        ]
        cov_prec[2][2] = minephi+np.sum([uncertainties[i][2][2] for i in range(idpair[1], idpair[0]+1)])#uncertainties[idpair[0]][2][2]+uncertainties[idpair[1]][2][2] # not so sure about phi
        
        d = get_motion_vector([0,0]+[pg.nodes[idpair[0]][2]], [0,0]+[pg.nodes[idpair[1]][2]])
        print "inserting loop closure vector btw. ",idpair,": ", d, np.rad2deg(d[2]), "\t theta uncertainty:", cov_prec[2][2]
        edge = [idpair[0], idpair[1], d, np.linalg.inv(cov_prec)]
        edges.append(edge)
        pg.edges.append(PoseEdge(*edge))
        
    writePoseGraph(pg.nodes, edges, 'data/goalpoints.graph') # write graph in g2o format for TORO
        
    # run SLAM
    plt.clf()
    plt.subplot(1,3,1)
    plt.title('path (before SLAM)')
    for j in range(len(path)):
        plt.scatter(pg.nodes[j, 0], pg.nodes[j, 1], s=10+200*(uncertainties[j][2][2]-minephi)/ephi)
        plt.text(pg.nodes[j, 0]+30*float(j)/len(path)+np.random.random()*2-1, pg.nodes[j, 1]+np.random.random()*2-1, str(j))
    c = 'rgbcmyk'  
    for e in loopedges:
        plt.scatter(pg.nodes[e[0], 0], pg.nodes[e[0], 1], s=80, c=c[np.mod(e[0],len(c))])
        plt.scatter(pg.nodes[e[1], 0], pg.nodes[e[1], 1], s=80, c=c[np.mod(e[0],len(c))])  
    print "optimization step (1)"
    pg.optimize(1)
    
    for k in range(2):
        plt.subplot(1,3,k+2)
        plt.title('path (after '+str(k+1)+' SLAM steps)')
        plt.scatter(path[:, 0], path[:, 1], s=1)
        plt.hold(True)

        for j in range(len(path)):
            plt.scatter(pg.nodes[j, 0], pg.nodes[j, 1], s=20+uncertainties[j][2][2]/minephi)
            plt.text(pg.nodes[j, 0], pg.nodes[j, 1], str(j))
        
        print "optimization step (10)"
        pg.optimize(10) 
        
    plt.show()


if __name__ == "__main__":
    import pickle
    with open('data/goalpointpath.b', 'rb') as f:
        coords = pickle.load(f)
    coords = np.array(coords)
    plt.scatter(coords[:, 0], coords[:,1])
    plt.title('ground truth')
    plt.show(block=True)
    #loopedges = getloops(coords)
    loopedges = [[20,8], [36,20], [98,36], [102,138], [112,128], [127,113]]
    runslam(coords, loopedges)