from matplotlib import pyplot as plt

times = [9.380, 5.433, 2.208]
timeSec = [t * 60 for t in times]

plt.figure(figsize=(8, 4))
plt.bar(['Trimesh', 'xSDF (CPU)', 'xSDF (GPU)'], times, width=0.3)
plt.ylabel('Time (s)')
plt.tight_layout()
plt.show()