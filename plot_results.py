import matplotlib.pyplot as plt
import numpy as np

# Suppose you stored your target and actual joint positions
target = np.load('target.npy')
actual = np.load('actual.npy')

plt.plot(target, label='Target')
plt.plot(actual, label='Actual')
plt.title('Trajectory Tracking Performance')
plt.xlabel('Timestep')
plt.ylabel('Joint Position')
plt.legend()
plt.show()
