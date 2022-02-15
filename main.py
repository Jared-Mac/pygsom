import GSOM
import numpy as np
import matplotlib.pyplot as plt


dataset = [[1, 0]]
dataset.append([0.5, 0])
dataset.append([.01, 0.01])
dataset.append([0.6, -0.06])

x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)
xvals = np.linspace(0, 2*np.pi, 50)
yinterp = np.interp(xvals, x, y)

plt.plot(x, y, 'o')
plt.plot(xvals, yinterp, '-x')
plt.show()
plt.ion()
# network = GSOM.GSOM(dataset,0.01,0.4)

# network.train()

# network.print()