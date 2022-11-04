import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
x = np.linspace(-10, 10, 10000)
y = np.maximum(0, x-1)+np.maximum(0, -1-x)
plt.axvline(0, color='k', linewidth=1)
plt.axhline(0, color='k', linewidth=1)
plt.plot(x, y)
plt.axvline(-1, color='r', linestyle='--', label='buffers')
plt.axvline(1, color='r', linestyle='--')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
# Set xticks at a point
ax.set_xticks([-1.075])

# Set xticklabels for the point
ax.set_xticklabels(["$\\bf{It\ is -\!1.075\ label}$"])
plt.show()
