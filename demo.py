import posegraph
from utils import readPoseGraph
reload(posegraph)

from posegraph import PoseGraph
import matplotlib.pyplot as plt

# Specify files to load
vfile = 'data/killian-v.dat'
efile = 'data/killian-e.dat'

pg = PoseGraph()
#pg.readGraph('data/goalpoints_toro_result.graph')
pg.readGraph(vfile, efile)

plt.ion()
plt.figure()
plt.scatter(pg.nodes[:, 0], pg.nodes[:, 1])
plt.show(block=True)

# Do 5 iteration with visualization
pg.optimize(5, plt)
#pg.optimize(5)

plt.show(block=True)