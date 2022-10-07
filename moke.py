import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)


fig, ax = plt.subplots()
for color in ['tab:blue', 'tab:orange', 'tab:green']:
    n = 750
    x, y = np.random.rand(2, n)
    scale = 200.0 * np.random.rand(n)
    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=1.0, edgecolors='none')

ax.legend()
ax.grid(True)

plt.show()
plt.savefig(f"/mnt/c/Users/chiaki/Desktop/mokemoke.png")