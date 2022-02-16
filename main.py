import GSOM
import numpy as np
import matplotlib.pyplot as plt





x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

x_total = x
y_total = y
dataset = list(zip(x,y))
for h in range(4):
    noise = np.random.normal(0, .035, y.shape)
    x_hat = x + noise
    np.append(x_total,x_hat)
    noise = np.random.normal(0, .035, y.shape)
    y_hat = y + noise
    np.append(y_total,y_hat)
    xy = list(zip(x_hat,y_hat))
    plt.plot(x_hat, y_hat, '.')
    dataset = dataset + xy

# dataset = list(zip(xvals,yinterp))
# dataset = list(zip(x,y))


network = GSOM.GSOM(dataset,0.92,0.5)

network.train()
network.print()

nodes_x = []
nodes_y = []
for node in network.nodes:
    nodes_x.append(node.array[0])
    nodes_y.append(node.array[1])

plt.plot(x_total, y_total, '.')

plt.plot(nodes_x,nodes_y,'x')
plt.show()
