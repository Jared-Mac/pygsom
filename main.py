import GSOM
import numpy as np
import matplotlib.pyplot as plt
import visualize
import networkx as nx 

G = nx.Graph()
G.add_node(GSOM.Node(0,1))



# x = np.linspace(0, 2*np.pi, 100)
# y = np.sin(x)

x = np.linspace(0,1,60)
y = np.sin(x)
y_p = np.cos(x)
print(y.shape)
x_total = x
y_total = y
dataset = list(zip(x,y))
dataset = dataset + list(zip(x,y_p))
for h in range(4):
    noise = np.random.normal(0, .015, y.shape)
    x_hat = x + noise
    np.append(x_total,x_hat)
    noise = np.random.normal(0, .015, y.shape)
    y_hat = y + noise
    np.append(y_total,y_hat)
    xy = list(zip(x_hat,y_hat))
    plt.plot(x_hat, y_hat, '.')
    dataset = dataset + xy



for h in range(4):
    noise = np.random.normal(0, .015, y_p.shape)
    x_hat = x + noise
    np.append(x_total,x_hat)
    noise = np.random.normal(0, .015, y_p.shape)
    y_hat = y_p + noise
    np.append(y_total,y_hat)
    xy = list(zip(x_hat,y_hat))
    plt.plot(x_hat, y_hat, '.')
    dataset = dataset + xy

# dataset = list(zip(xvals,yinterp))
# dataset = list(zip(x,y))
np.random.shuffle(dataset)
# print(dataset)

basis = GSOM.GSOM(dataset,0.93,0.15)

basis.train()
print(basis.averageError())

# origin = np.array([0,0])
# basis_nodes = []
# for node in basis.nodes:
#     basis_nodes.append((node,GSOM.distance(node.array, origin)))

# basis_nodes = sorted(basis_nodes,key=GSOM.getKey)
# print(basis_nodes)
    



# network = GSOM.GSOM(dataset, 0.92,0.40,connected=True,basis = basis_nodes)
# network.train()
nodes_x = []
nodes_y = []
basis_x = []
basis_y = []
for node in basis.connectNodes():
    nodes_x.append(node.array[0])
    nodes_y.append(node.array[1])
# for node in network.basis:
#     basis_x.append(node.array[0])
#     basis_y.append(node.array[1])

plt.plot(x_total, y_total, '.')
plt.plot(nodes_x,nodes_y,'-o', markersize=10)
# plt.plot(basis_x,basis_y,'-o', markersize=10)


plt.show()
