import posegraph
reload(posegraph)

from posegraph import PoseGraph
import matplotlib.pyplot as plt

# Specify files to load
vfile = 'data/killian-v.dat'
efile = 'data/killian-e.dat'

#plt.ion()
#plt.figure()
#plt.show()

pg = PoseGraph()
pg.readGraph(vfile, efile)
# Do 5 iteration with visualization
#pg.optimize(5, plt)
pg.optimize(5)

plt.show(block=True)