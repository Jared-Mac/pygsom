from audioop import reverse
from email.policy import default
from tkinter.tix import Tree
import uuid
import numpy as np
import networkx as nx 
import itertools
import matplotlib.pyplot as plt
from matplotlib.path import Path


import matplotlib.cm as cm
from uuid import uuid1
from dataclasses import dataclass,field
import timeit 
import numba as nb
@nb.njit(fastmath=True)
def norm(l):
    s = 0.
    for i in range(l.shape[0]):
        s += l[i]**2
    return np.sqrt(s)
def convertGraph(input):

    network = nx.Graph()

    
    added = {}

    i = 0
    for u,v,data in input.edges(keys=False,data=True):
        i += 1
        if u not in added:
            n1 = Node(np.array([input.nodes[u]['y'],input.nodes[u]['x']], dtype=float))
            network.add_node(n1)
            added[u] = n1
        if v not in added:
            n2 = Node(np.array([input.nodes[v]['y'],input.nodes[v]['x']], dtype=float))
            network.add_node(n2)
            added[v] = n2
        
        polyline = list(data["geometry"].coords)

        # Construct new node and edge
        weights = np.array([polyline[0][1],polyline[0][0]],dtype=float)
        prev_node = Node(weights)
        network.add_node(prev_node)

        network.add_edge(added[v],prev_node,weight=1)


        for pair in list(data["geometry"].coords[2:-1]):
            weights = np.array([pair[1],pair[0]],dtype=float)
            new_node = Node(weights)
            network.add_node(new_node)
            network.add_edge(new_node,prev_node,weight=1)  
            prev_node = new_node
        

        network.add_edge(prev_node,added[u],weight=1)  


    # Clean Data to inside bounding box
    # 32.93127,32.92748,-117.17359,-117.17801
    min_lat = 32.92748
    max_lat = 32.93127
    min_long = -117.17359
    max_long = -117.17801

    for node in list(network.nodes):
        lat = node.array[0]
        long = node.array[1]
        remove = False
        if lat < min_lat or lat > max_lat:
            remove = True
        if abs(long) < abs(min_long) or abs(long) > abs(max_long):
            remove = True
        if remove:
            network.remove_node(node)

            

    pos_dict = {}
    for node in network:
        pos_dict[node] = node.array
    # edges,weights = zip(*nx.get_edge_attributes(network,).items())
    # nx.draw(network, edgelist=edges,pos=pos_dict,node_size=50)
    plt.show()

    return network
def findSegments(graph):
    paths = dict(nx.all_pairs_shortest_path(graph))
    segments = []
    count = 0
    nodes_j = []
    for node,degree in list(graph.degree()):
        if degree == 1 or degree > 2:
            nodes_j.append(node)

    for x1,x2 in itertools.combinations(nodes_j,2):
        flag = True
        if nx.has_path(graph,x1,x2):
            for y in paths[x1][x2][1:-1]:
                for z in nodes_j:
                    if z == y:
                        flag = False
                        break
            if flag:
                segments.append(paths[x1][x2])
            count += 1
    print(f"amount of segments {len(segments)}")
    return segments
def distance(n1,n2):
    return np.linalg.norm(n1.array-n2.array,ord=3)
def gpsDistance(n1,n2):
    return norm(n1-n2)
    return np.linalg.norm(n1-n2)

def generateID() -> int:
    return uuid.uuid1()

def distanceToLine(point,n1,n2):
    x_0, y_0 = point[0], point[1]
    x_1,y_1 = n1[0],n1[1]
    x_2,y_2 = n2[0],n2[1]
    return (np.abs((x_2 - x_1)*(y_1 - y_0) - (x_1 - x_0)*(y_2 - y_1)))/(np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2))
def middleNode(n1,n2):
    x = (n1.array[0] + n2.array[0])/2
    y = (n1.array[1] + n2.array[1])/2
    return Node(x,y)
def getKey(item):
    return item[1]

# def findSegments(graph):
#     # Find Shortest Paths

#     # Find Intersection Nodes 

#     # Cut paths on intersection

#     # Return 
#     return

@dataclass
class Node:
    array: np.ndarray
    id: int = field(init=False,default_factory=generateID)
    error: float = 0
    def __hash__(self): return hash(self.id)
    def addError(self,newError):
        self.error =+ newError
    def __eq__(self,obj):
        return self.id == obj.id

class GSOM:

    def __init__(self,dataset,spread_factor,radius,connected=False,basis=None):
        self.network = nx.Graph()
        # Starting Learning Rate
        self.spread_factor = spread_factor
        # growth_threshold = -2 * ln(self.spread_factor)
        self.growth_threshold = -2 * np.log(self.spread_factor)
        print(f"Growth Threshold = {self.growth_threshold}")
        self.data = dataset
        self.iteration = 0
        self.radius = radius
        self.addStartNodes()
        self.connected = connected
        self.smooth = False
        self.cleanData()
        self.weight_change = 0

        if connected:
            self.basis = []
            for pair in basis:
                self.basis.append(pair[0])
            self.createClosestBasis()
    def addNode(self,Node):
        self.network.add_node(Node)
    def addStartNodes(self):
        # n1 = Node(0,0)
        # n2 = Node(0.5,0.5)
        # n3 = Node(1,1)
        
        n1 = Node(np.array([self.data[0][0],self.data[0][1]]))
        n2 = Node(np.array([self.data[1][0],self.data[1][1]]))
        n3 = Node(np.array([self.data[2][0],self.data[2][1]]))
        n1.right = n2
        n2.left, n2.right = n1,n3
        n3.left = n2
        self.addNode(n1)
        self.addNode(n2)
        self.addNode(n3)

    def getNodes(self):
        return self.network.nodes(data=True)
    def getNetworkSize(self):
        return self.network.number_of_nodes()
    def learningRate(self):
        # learning rate = at iteration k, with r nodes, initial learning rate lr_0
        # d is constant, how much the learning rate should drop
        # lr_0*d^floor((1+k)/r)
        return 1*(0.02**(1+self.iteration)/self.getNetworkSize())
    def findWinner(self,input):
        error=float("inf")
        winner = None
        for node in self.network:
            # starttime = timeit.default_timer()
            d = gpsDistance(input,node.array)
            # print("The time difference is :", timeit.default_timer() - starttime)
            if(d < error):
                error = d
                winner = node
        return winner, error
    
    def findWinners(self,input,num_winners):
        node_list = []
        for node in self.network:
            # starttime = timeit.default_timer()
            node_list.append([node,gpsDistance(input,node.array)])
            # print("The time difference is :", timeit.default_timer() - starttime)
        node_list = sorted(node_list,key=getKey)

        return node_list[:num_winners]

    def findTopTwo(self,input):

        node_list = []
        for node in self.network:
            node_list.append((node,gpsDistance(input,node.array)))
        node_list = sorted(node_list,key=getKey)

        return (node_list[0],node_list[1])
    def adaptCollapseNode(self,node):
        radius = self.radius * 1.5
        for x in range(5):
            for inputData in self.basisInputNeighborhood(node,radius):
                node.array = node.array + 0.01 * gpsDistance(node.array,inputData)

    
    def neighborhood(self,focus,radius):
       
        neighborhood = []
        for node in self.network:
            if gpsDistance(node.array,focus.array) < radius:
                neighborhood.append(node)
        return neighborhood
    def basisInputNeighborhood(self,focus,radius):
    
        inputNeighborhood = []
        for inputData in self.data:
            if gpsDistance(inputData,focus.array) < radius:
                inputNeighborhood.append(inputData)
        return inputNeighborhood
    
    def averageError(self):
        error = 0
        for node in self.network:
            error += node.error
        return error / self.getNetworkSize()
    def maxError(self):
        error = 0
        for node in self.network:
            if node.error > error:
                error = node.error
        return error
    def calculateLinkScore(self,n1,n2):
        p1 = n1.array
        p2 = n2.array
        r = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        d = 0.04
        x_delta = d/r * (p1[1] - p2[1])
        y_delta = d/r * (p1[0] - p2[0])

        p3 = (p1[0] + x_delta,p1[1] - y_delta)
        p4 = (p1[0] - x_delta,p1[1] + y_delta)

        p5 = (p2[0] + x_delta,p2[1] - y_delta)
        p6 = (p2[0] - x_delta,p2[1] + y_delta)

        vertices = [p3,p4,p6,p5,p3]
        path = Path(vertices)
        hits = path.contains_points(self.data)
        count  = np.count_nonzero(hits)

        return count / (x_delta*y_delta)


    def createClosestBasis(self):
        new_basis = [self.basis[0]]
        unseen = self.basis[1:]
        last_added = self.basis[0]
        while len(new_basis) != len(self.basis):
            node_list = []
            for node in unseen:
                node_list.append((node,gpsDistance(node.array,last_added.array)))
            node_list = sorted(node_list,key=getKey)
            new_node = node_list[0][0]
            # print(unseen)
            # print(new_node)
            unseen.remove(new_node)
            new_basis.append(new_node)
            last_added = new_node

        self.basis = new_basis
    def connectNodes(self):
        # nodes = self.network.nodes()
        
        for datapoint in self.data:
            # find top 2 winnings nodes
            winner1,winner2 = self.findWinners(datapoint, 2)
            # add edge
            self.network.add_edge(winner1[0],winner2[0],weight=1)
            # distance = float("inf")
            # winner1,winner2 = None,None
            # for n1,n2 in itertools.combinations(winning_nodes,2):
            #     d = distanceToLine(datapoint,n1[0].array,n2[0].array)
            #     if d < distance:
            #         distance = d
            #         winner1, winner2 = n1[0],n2[0]
            # if distance < self.radius * 1.5:
            #     self.network.add_edge(winner1,winner2,weight=1)

            

    def connectRelativeNeighborhood(self):
        for u,v in itertools.combinations(list(self.network.nodes),2):
            new_list = list(self.network.nodes)
            new_list.remove(u)
            new_list.remove(v)
            connect  = True
            for z in new_list:
                base_distance = distance(u,v)
                if distance(z,u) < base_distance and distance(z,v) < base_distance:
                    connect = False
                    break
            if connect:
                self.network.add_edge(u,v,weight=1)


    def adaptWeights(self,winning_node,input):
        # Adapt weight of winner and nodes within neighborhood
        #   Weights are distributed proportionally to neighborhood nodes by distance to winner
        # N_k+1 is the neighborhood of winning neuron at k+1 iteration
        # LR(k) should depend on number of nodes at time k
        # w_j(k+1) = w_j(k)                          if j not in N_k+1
        #            w_j(k) + LR(k)*(x_k-w_j(k))     if j in N_k+1
        # where LR(k) is learning rate of K in N, sequence of positive parameters converging to 0 as k->infinity
        # w_j(k),w_j(k+1) are nodes, j, before and after adaption

        for node in self.neighborhood(winning_node,self.radius):
            weight_update = self.learningRate() * gpsDistance(node.array,input)
            self.weight_change += weight_update
            node.array = node.array + weight_update
    def growNode(self,winning_node,input):

        new_node = Node(np.array(input))
        self.addNode(new_node)

        winning_node.error = 0
    def collapse(self):
        last_node_pool = []
        while(sum(nx.triangles(self.network).values()) / 3 >= 1):
            inv_map = {}
            for k, v in nx.triangles(self.network).items():
                inv_map[v] = inv_map.get(v, []) + [k]
            
            degrees = inv_map.keys()

            max_deg = max(inv_map.keys())
            node_pool = []
            for key in degrees:
                node_pool = node_pool + inv_map[key]

            breakFlag = False
            vertex = inv_map[max_deg][0]

            vertices = []
            for x in range(max_deg):
                for node1,node2 in itertools.combinations(node_pool,2):
                    if self.network.has_edge(node1,node2) and self.network.has_edge(node1,vertex) and self.network.has_edge(vertex,node2):
                        print(f"Collapsing Triangle")

                        if node1 in vertices:
                            pass
                        else:
                            vertices.append(node1)

                        if node2 in vertices:
                            pass
                        else:
                            vertices.append(node2)
                        breakFlag = True
                        break
                if breakFlag:
                    continue
            sum_vertices = np.array([0,0])
            for x in vertices:
                sum_vertices = sum_vertices + x.array
                self.network = nx.contracted_nodes(self.network,vertex,x,self_loops=False)
            vertex.array = sum_vertices / len(vertices)
            
                    

            
    def growing(self):
        # Iterate until terminal condition
        # Grab data point
        
        average_error = float("inf")
        last_avg = average_error
        max_error = float("inf")

        for x in range(200):
        # while error_change > 0.01:
            self.weight_change = float(0)
            for inputData in self.data:
                
                # print(input)
                # print(f"Iteration #: {self.iteration}")
            
                winning_node,error = self.findWinner(inputData)
                self.adaptWeights(winning_node,inputData)


                # Increase error value of winner (difference between input and node)
                # When error value > growth threshold, grow node if winner is a boundary node
                # Otherwise  distribute weights

                winning_node.addError(error)

                # if winning_node.error > self.growth_threshold:
                #     self.growNode(winning_node,inputData)
                if gpsDistance(winning_node.array,inputData) > self.growth_threshold:
                    self.growNode(winning_node,inputData)


                self.iteration += 1
            # self.mergeNodes()
            
            average_error = self.averageError()
            max_error = self.maxError()
            

            print(f"Weight Change {self.weight_change}")
            print(f"Average error {average_error}")            
            print(f"Max error {max_error}")
            if self.weight_change <0.01:
                print("Weights didn't change")
                return


    def growingConnected(self):
        for x in range(50):
            for basis_node in self.basis:
                for y in range(5):
                    for inputData in self.basisInputNeighborhood(basis_node,self.radius):
                        winning_node,error = self.findWinner(inputData)
                        self.adaptWeights(winning_node,inputData)


                        # Increase error value of winner (difference between input and node)
                        # When error value > growth threshold, grow node if winner is a boundary node
                        # Otherwise  distribute weights

                        winning_node.addError(error)

                        if winning_node.error > self.growth_threshold:
                            self.growNode(winning_node,inputData)

                        self.iteration += 1                      
        # Reset LR to start value

    def smoothing(self):
        # Reduce learning rate and decrease neighborhood
        # Find winners and adapt weights as in growing phase
        self.smooth = True
        self.radius = self.radius * 2
        for i in range(50):
            self.weight_change = 0
            for inputData in self.data:
                winning_node, error = self.findWinner(inputData)
                self.adaptWeights(winning_node, inputData)
                self.iteration += 1 
            print(f"Weight Change {self.weight_change}")
            if self.weight_change < 0.01:
                print("Weights didn't change")
                break


        self.radius = self.radius / 3
        for i in range(20):
            self.weight_change = 0  
            for inputData in self.data:
                winning_node, error = self.findWinner(inputData)
                self.adaptWeights(winning_node, inputData)
                self.iteration += 1 
            print(f"Weight Change {self.weight_change}")
            if self.weight_change < 0.01:
                print("Weights didn't change")
                break


        
    def train(self):
        if self.connected:
            self.growingConnected()
        else:
            self.growing()
        self.connectNodes()
        # self.connectRelativeNeighborhood()
        self.collapse()
        self.smoothing()
        self.scaleGraph(1000)
        return
    def adjacencyMatrix(self):
        return nx.convert_matrix.to_numpy_array(self.network)
    def print(self):
        # print(list(self.network.nodes(data=True)))
        print(f"Triangles: {sum(nx.triangles(self.network).values()) / 3}")
    def cleanData(self):
        new_data = []
        for datapoint in self.data:
            if len(self.basisInputNeighborhood(Node(datapoint),self.radius)) > 1:
                new_data.append(datapoint)
        self.data = new_data
    def scaleGraph(self,scaling_factor):
        for node in self.network:
            node.array = node.array / scaling_factor
            node.x = node.array[0]
            node.y = node.array[1]
    
    def getSegments(network):
        pass
    def getNodes_h(network):
        nodes_h = []
        for node,degree in list(self.network.degree()):
            if degree == 0 or degree> 2:
                nodes_h.append(node)
        return

    def mergeNodes(self):
        pass
        nodes = list(self.network)
        for node in nodes:
            neighbors = self.neighborhood(node,self.radius / 2)
            for neighbor in neighbors:
                if gpsDistance(node.array,neighbor.array) < self.radius / 2:
                    x = (node.array[0] + neighbor.array[0]) / 2
                    y = (node.array[1] + neighbor.array[1]) / 2
                    self.addNode(Node(x,y))
                    try:
                        self.network.remove_node(node)
                        self.network.remove_node(neighbor)
                    except nx.exception.NetworkXError:
                        pass


    def completeness():
        for edge_c in self.network.Edges():
            for edge_t in ground_truth:
                n = 0
                distanceSum = 0
                for point_c in s_c:
                    distanceSum += distance(point_c,edge_t)
                    n += 1
                # distance  edge_c & edge_t
            # flag edge_t with minimum distance to edge_c as matched segment edge_t_h
        # l = total length of matched segments of edge_t_h
        # L = total length of segments edge_t in ground truth network
        # return l / L            
        return 
    def precision():
        m = 0
        for edge_c in self.network.Edges():
            for edge_t in ground_truth:
                n = 0
                distanceSum = 0
                for point_c in s_c:
                    distanceSum += distance(point_c,edge_t)
                    n += 1
            #  edge_t_h = edge_t with minimum distance to edge_c
            # m += 1
            # precisionArray[m] = distance between edge_c and edge_t_h
        # return average and std of precisionArray
        return
    def topology():
        # A_t = Floyd(A_t)
        # A_c = Floyd(A_c)
        # a_t = average of non-diagonal elements (A_t)
        # a_c =  average of non-diagonal elements (A_c)
        # return a_t / a_c
        return
    def compareNetwork(self):
        
        self.completeness()
        self.precision()
        self.topology()



    def visualizeGraph(self):
        pos_dict = {}
        for node in self.network:
            pos_dict[node] = node.array
        edges,weights = zip(*nx.get_edge_attributes(self.network,'weight').items())
        # nx.draw(self.network, edgelist=edges, edge_color=weights,pos=pos_dict,edge_cmap=cm.Reds,node_size=50)
        nx.draw(self.network, edgelist=edges,pos=pos_dict,node_size=50)
        # plt.show()

