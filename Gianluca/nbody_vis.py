import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation
import pandas as pd
import os, sys

NUM_BODIES = 1024
NUM_ITER = 5000
MAX_POS = 12500

t = np.array([np.ones(NUM_BODIES)*i for i in range(NUM_ITER)]).flatten()
x = np.zeros(shape=NUM_BODIES*NUM_ITER, dtype=np.float32, order='C')
y = np.zeros(shape=NUM_BODIES*NUM_ITER, dtype=np.float32, order='C')
z = np.zeros(shape=NUM_BODIES*NUM_ITER, dtype=np.float32, order='C')


filename = "out.csv"
with open(filename, 'r') as f:
	num = 0
	for line in f.readlines():
		x[num], y[num], z[num] = np.array(line.rstrip('\n').split(','), dtype=np.float32)
		num += 1

df = pd.DataFrame({"time": t, "x": x, "y": y, "z": z})

def init():
    ax.set_xlim(-1*MAX_POS, MAX_POS)
    ax.set_ylim(-1*MAX_POS, MAX_POS)
    return graph, 

def update_graph(num):
	data=df[df['time']==num]
	graph.set_data (data.x, data.y)
	graph.set_3d_properties(data.z)
	return graph, 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Test')
ax.set_xlim(-1*MAX_POS, MAX_POS)
ax.set_ylim(-1*MAX_POS, MAX_POS)
ax.set_zlim(-1*MAX_POS, MAX_POS)

data=df[df['time']==0]
graph, = ax.plot(data.x, data.y, data.z, linestyle="", marker="o")

ani = matplotlib.animation.FuncAnimation(
	fig=fig, func=update_graph, frames=NUM_ITER-1, init_func=init, interval=1, blit=False)

plt.show()

