import posegraph
import time
from utils import *
reload(posegraph)

from posegraph import PoseGraph
import matplotlib.pyplot as plt

# Specify files to load
vfile = 'data/killian-v.dat'
efile = 'data/killian-e.dat'

pg = PoseGraph()
pg.readGraph(vfile, efile)

"""
pg.readGraph('data/goalpoints_toro_result.graph')
cols = [(np.random.random(), np.random.random(), np.random.random()) for i in range(1000)]
coords = pg.nodes[:, :2]
uncertainties, path = get_uncertainties_and_path(coords)
for j in range(len(path)):
    plt.scatter(pg.nodes[j, 0], pg.nodes[j, 1], s=1+np.min((200, 1000*(uncertainties[j][2][2])/ephi**2)))
for e in [[20,8], [36,20], [98,36], [102,138], [112,128], [127,113]]:
    plt.scatter(pg.nodes[e[0], 0], pg.nodes[e[0], 1], s=80, c=cols[e[0]])
    plt.text(pg.nodes[e[0], 0]+(np.random.random()*1-.5), pg.nodes[e[0], 1]+(np.random.random()*1-.5), str(e[0]))
    plt.scatter(pg.nodes[e[1], 0], pg.nodes[e[1], 1], s=80, c=cols[e[0]])
    plt.text(pg.nodes[e[1], 0]+(np.random.random()*1-.5), pg.nodes[e[1], 1]+(np.random.random()*1-.5), str(e[1]))
plt.show()
"""

plt.ion()
plt.figure()
plt.scatter(pg.nodes[:, 0], pg.nodes[:, 1])
plt.draw()
time.sleep(1)

# Do 5 iteration with visualization
pg.optimize(5, plt)

plt.show(block=True)