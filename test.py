import GSOM
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx 

def sin_cos_dataset():
    x = np.linspace(0,1,60)
    y = np.sin(x)
    y_p = np.cos(x)
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

        dataset = dataset + xy
    for h in range(4):
        noise = np.random.normal(0, .015, y_p.shape)
        x_hat = x + noise
        np.append(x_total,x_hat)
        noise = np.random.normal(0, .015, y_p.shape)
        y_hat = y_p + noise
        np.append(y_total,y_hat)
        xy = list(zip(x_hat,y_hat))

        dataset = dataset + xy
    return dataset





dataset = np.load("data\small_scale.npy")
dataset = dataset * 1000
# dataset = sin_cos_dataset()
np.random.shuffle(dataset)






basis = GSOM.GSOM(dataset,0.925,0.10)

start_time = time.perf_counter() 
basis.train()
end_time = time.perf_counter()
print(f'Time training: {end_time-start_time}')


data = np.array(basis.data)

x_data = data[:,[0]] / 1000
y_data = data[:,[1]] / 1000
print(basis.network)
plt.plot(x_data, y_data, '.', markerfacecolor='red')


print(f"Triangles: {sum(nx.triangles(basis.network).values()) /3}")
basis.visualizeGraph()

plt.show()