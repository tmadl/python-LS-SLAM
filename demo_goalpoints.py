import time
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from posegraph import PoseGraph, PoseEdge

cols = [(np.random.random(), np.random.random(), np.random.random()) for i in range(1000)]

coords = []
loopedges = []

def runslam(coords, loopedges):
    uncertainties, path = get_uncertainties_and_path(coords, linear_uncertainty=ex, angular_uncertainty=ephi)
    
    pg = PoseGraph() # LS-SLAM pose graph
    edges = []
    # populate with data -  path
    for p in range(len(path)):
        pg.nodes.append(path[p])
        if p > 0:
            covm = uncertainties[p]#+uncertainties[p-1]
            try:
                infm=np.linalg.inv(covm)
            except:
                infm=np.linalg.inv(covm)
            edge = [p-1, p, get_motion_vector(path[p], path[p-1]), infm]
            edges.append(edge)
            pg.edges.append(PoseEdge(*edge))
    pg.nodes = np.array(pg.nodes)
    # populate with data - loop closures
    for idpair in loopedges:
        sx = minex
        sy = miney   
        st = minephi+ephi*np.sum([np.abs(path[p][2]-path[p-1][2]) for p in range(idpair[1], idpair[0]+1)])
        cov_prec = [
        [sx**2,0,0],
        [0,sy**2,0],
        [0,0,st**2]
        ]
        
        d = get_motion_vector([0,0]+[pg.nodes[idpair[0]][2]], [0,0]+[pg.nodes[idpair[1]][2]])
        print "inserting loop closure vector btw. ",idpair,": ", d, np.rad2deg(d[2]), "\t theta uncertainty:", cov_prec[2][2]
        try:
            edge = [idpair[0], idpair[1], d, np.linalg.inv(cov_prec)]
        except:
            edge = [idpair[0], idpair[1], d, np.linalg.inv(cov_prec)]
        edges.append(edge)
        pg.edges.append(PoseEdge(*edge))
        
    writePoseGraph(pg.nodes, edges, 'data/goalpoints.graph') # write graph in g2o format for TORO
        
    # run SLAM
    plt.clf()
    #plt.subplot(1,3,1)
    plt.title('path (before SLAM)')
    for j in range(len(path)):
        plt.scatter(pg.nodes[j, 0], pg.nodes[j, 1], s=1+np.min((200, 1000*(uncertainties[j][2][2])/ephi**2)))
        #plt.text(pg.nodes[j, 0]+30*float(j)/len(path)+np.random.random()*2-1, pg.nodes[j, 1]+np.random.random()*2-1, str(j))
    for e in loopedges:
        plt.scatter(pg.nodes[e[0], 0], pg.nodes[e[0], 1], s=80, c=cols[e[0]])
        plt.text(pg.nodes[e[0], 0]+(np.random.random()*1-.5), pg.nodes[e[0], 1]+(np.random.random()*1-.5), str(e[0]))
        plt.scatter(pg.nodes[e[1], 0], pg.nodes[e[1], 1], s=80, c=cols[e[0]])
        plt.text(pg.nodes[e[1], 0]+(np.random.random()*1-.5), pg.nodes[e[1], 1]+(np.random.random()*1-.5), str(e[1]))
    plt.draw()
    time.sleep(1)
    print "optimization step (1)"
    pg.optimize(1)
    
    for k in range(2):
        plt.subplot(1,2,k+1)
        plt.title('path (after '+str(k+1)+' SLAM steps)')
        plt.scatter(path[:, 0], path[:, 1], s=1)
        plt.hold(True)

        for j in range(len(path)):
            plt.scatter(pg.nodes[j, 0], pg.nodes[j, 1], s=1+np.min((200, 1000*(uncertainties[j][2][2])/ephi**2)))
            #plt.text(pg.nodes[j, 0], pg.nodes[j, 1], str(j))
        plt.draw()
        print "optimization step (5)"
        pg.optimize(5) 
        
    plt.show(block=True)


if __name__ == "__main__":
    import pickle
    with open('data/goalpointpath.b', 'rb') as f:
        coords = pickle.load(f)
    coords = np.array(coords)
    plt.ion()
    plt.figure()
    plt.show()
    plt.scatter(coords[:, 0], coords[:,1])
    plt.title('ground truth')
    plt.draw()
    time.sleep(1)
    #loopedges = getloops(coords)
    loopedges = [[20,8], [36,20], [98,36], [102,138], [112,128], [127,113]]
    loopedges.reverse()
    runslam(coords, loopedges)