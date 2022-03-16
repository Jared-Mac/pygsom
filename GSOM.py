from audioop import reverse
from tkinter.tix import Tree
import numpy as np
import networkx as nx 
import itertools
import matplotlib.pyplot as plt
from matplotlib.path import Path
import geopy.distance
import matplotlib.cm as cm


def distance(n1,n2):
    return np.linalg.norm(n1-n2)
def gpsDistance(n1,n2):
    return np.linalg.norm(n1-n2)
    return geopy.distance.distance(n1,n2).m
def middleNode(n1,n2):
    x = (n1.array[0] + n2.array[0])/2
    y = (n1.array[1] + n2.array[1])/2
    return Node(x,y)
def getKey(item):
    return item[1]
  
class Node:
    def __init__(self, x,y):
        """Initialize a node object with values x,y"""
        self.x, self.y = x, y
        self.array = np.array([x,y])
        self.error = 0
        self.left = None
        self.right = None
    
    def addError(self,newError):
        self.error =+ newError

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

        n1 = Node(self.data[0][0],self.data[0][1])
        n2 = Node(self.data[1][0],self.data[1][1])
        n3 = Node(self.data[2][0],self.data[2][1])
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
        if self.smoothing:
            return 0.5*(0.01**np.floor((1+self.iteration)/self.getNetworkSize()))
        return 1*(0.02**np.floor((1+self.iteration)/self.getNetworkSize()))
    def findWinner(self,input):
        error=float("inf")
        winner = None
        for node in self.network:
            d = gpsDistance(input,node.array)
            if(d < error):
                error = d
                winner = node
        return winner, error

    def findTopTwo(self,input):

        node_list = []
        for node in self.network:
            node_list.append((node,gpsDistance(input,node.array)))
        node_list = sorted(node_list,key=getKey)

        return (node_list[0],node_list[1])

    
    def neighborhood(self,focus,radius):
       
        if self.smooth:
            radius = radius / 3
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
        nodes = self.network.nodes()
        for datapoint in self.data:
            # find top 2 winnings nodes
            winner1,winner2 = self.findTopTwo(datapoint)
            # add edge

            self.network.add_edge(winner1[0],winner2[0],weight=1)

        # 
        # Iterates through each pair of nodes
        # scores = []
        # for n1,n2 in itertools.combinations(nodes,2):
        #     score = self.calculateLinkScore(n1,n2)
        #     scores.append(score)
        #     if score > 1:
        #         self.network.add_edge(n1,n2,weight=score)
        # scores.sort(reverse=True)  
        # print(scores)
    
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
            node.array = node.array + self.learningRate() * gpsDistance(node.array,input)
    def growNode(self,winning_node,input):
        if self.connected:
            if winning_node.left != None and winning_node.right != None:
                input_node = Node(input[0],input[1])
                new_node = middleNode(winning_node,input_node)
                if distance(winning_node.left.array, input_node.array) < gpsDistance(winning_node.right.array, input_node.array):
                    # Add new node to winning_nodes left
                    # winning_node.left --- new_node --- winning_node
                    winning_node.left.right = new_node
                    new_node.right = winning_node
                    new_node.left = winning_node.left
                    winning_node.left = new_node
                    
                    self.addNode(new_node)
                else: 
                    # winning_node --- new_node --- winning_node.right
                    winning_node.right.left = new_node
                    new_node.left = winning_node
                    winning_node.right = new_node
                    new_node.left = winning_node
                    self.addNode(new_node)
                # pass
            elif winning_node.left == None:
                new_node = Node(input[0],input[1])
                winning_node.left = new_node
                new_node.right = winning_node
                self.addNode(new_node)
            elif winning_node.right == None:
                new_node = Node(input[0],input[1])
                winning_node.right = new_node
                new_node.left = winning_node
                self.addNode(new_node)
        else: 
            new_node = Node(input[0],input[1])
            self.addNode(new_node)

        winning_node.error = 0

    def growing(self):
        # Iterate until terminal condition
        # Grab data point
        average_error = float("inf")
        max_error = float("inf")
        # for x in range(200):
        while average_error > 0.07 and max_error > 0.10:
            for inputData in self.data:
                # print(input)
                # print(f"Iteration #: {self.iteration}")
            
                winning_node,error = self.findWinner(inputData)
                self.adaptWeights(winning_node,inputData)


                # Increase error value of winner (difference between input and node)
                # When error value > growth threshold, grow node if winner is a boundary node
                # Otherwise  distribute weights

                winning_node.addError(error)

                if winning_node.error > self.growth_threshold:
                    self.growNode(winning_node,inputData)

                self.iteration += 1
            self.mergeNodes()
            average_error = self.averageError()
            print(f"Average error {average_error}")            
            max_error = self.maxError()
            print(f"Max error {max_error}")

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
        self.radius = self.radius / 2
        for i in range(20):
           for inputData in self.data:
                # print(input)
                # print(f"Iteration #: {self.iteration}")

                winning_node, error = self.findWinner(inputData)
                self.adaptWeights(winning_node, inputData)


                self.iteration += 1 
                # average_error = self.averageError()
                # print(f"Average error {average_error}")
                # max_error = self.maxError()
                # print(f"Max error {average_error}")
        self.mergeNodes()
        self.connectNodes()
        print(self.network.size())
        
    def train(self):
        if self.connected:
            self.growingConnected()
        else:
            self.growing()
        self.smoothing()
        self.scaleGraph(1000)
    def adjacencyMatrix(self):
        return nx.convert_matrix.to_numpy_array(self.network)
    def print(self):
        return list(self.network.nodes(data=True))
    def cleanData(self):
        new_data = []
        for datapoint in self.data:
            if len(self.basisInputNeighborhood(Node(datapoint[0],datapoint[1]),self.radius)) > 1:
                new_data.append(datapoint)
        self.data = new_data
    def scaleGraph(self,scaling_factor):
        for node in self.network:
            node.array = node.array / scaling_factor
            node.x = node.array[0]
            node.y = node.array[1]
    def mergeNodes(self):
        nodes = list(self.network)
        for node in nodes:
            neighbors = self.neighborhood(node,self.radius)
            for neighbor in neighbors:
                if gpsDistance(node.array,neighbor.array) < self.radius / 2:
                    x = (node.array[0] + neighbor.array[0]) / 2
                    y = (node.array[1] + neighbor.array[1]) / 2
                    self.addNode(Node(x,y))
                    try:
                        self.network.remove_node(node)
                        self.network.remove_node(neighbor)
                    except nx.exception.NetworkXError:
                        print("Node already removed")

    def visualizeGraph(self):
        pos_dict = {}
        for node in self.network:
            pos_dict[node] = node.array
        edges,weights = zip(*nx.get_edge_attributes(self.network,'weight').items())
        # nx.draw(self.network, edgelist=edges, edge_color=weights,pos=pos_dict,edge_cmap=cm.Reds,node_size=50)
        nx.draw(self.network, edgelist=edges,pos=pos_dict,node_size=50)
        plt.show()

