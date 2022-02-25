import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def show_graph(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    x = 0
    for n in all_rows:
        gr.add_node(n,pos=(x,x))
        x += 1
    gr.add_edges_from(edges)
    nx.draw(gr, nx.get_node_attributes(gr, 'pos'), node_size=900)
    print(gr)
    plt.show()



