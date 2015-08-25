import posegraph
reload(posegraph)

from posegraph import PoseGraph
import matplotlib.pyplot as plt

import geometry

from utils import *

# Specify files to load
vfile = 'data/killian-v.dat'
efile = 'data/killian-e.dat'

pg = PoseGraph()
pg.readGraph(vfile, efile)

N = 100
plt.scatter(pg.nodes[:N, 0], pg.nodes[:N, 1])
plt.hold(True)

for i in range(N):
    f = pg.edges[i].id_from
    t = pg.edges[i].id_to
    #plt.plot([pg.nodes[f, 0], pg.nodes[t, 0]], [pg.nodes[f, 1], pg.nodes[t, 1]])
    
    print "p1 ", pg.nodes[f, :]
    print "p2 ", pg.nodes[t, :]
    print "edge ", pg.edges[i].mean
    
    Tp1 = t2v(np.dot(v2t(pg.nodes[f, :]), v2t(pg.edges[i].mean)))
    plt.plot([pg.nodes[f, 0], Tp1[0]], [pg.nodes[f, 1], Tp1[1]])

plt.show(block=True)