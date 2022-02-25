from platform import node
import numpy as np
from sklearn.metrics import euclidean_distances

def distance(n1,n2):
    return np.linalg.norm(n1-n2)
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
        self.nodes.append(n1)
        self.nodes.append(n2)
        self.nodes.append(n3)

    def __init__(self,dataset,spread_factor,radius,connected=False,basis=None):
        # Starting Learning Rate
        self.spread_factor = spread_factor
        # growth_threshold = -2 * ln(self.spread_factor)
        self.growth_threshold = -2 * np.log(self.spread_factor)
        print(f"Growth Threshold = {self.growth_threshold}")
        self.nodes = []
        self.data = dataset
        self.iteration = 0
        self.radius = radius
        self.addStartNodes()
        self.connected = connected
        self.smooth = False

        if connected:
            self.basis = []
            for pair in basis:
                self.basis.append(pair[0])
            self.createClosestBasis()

    def learningRate(self):
        # learning rate = at iteration k, with r nodes, initial learning rate lr_0
        # d is constant, how much the learning rate should drop
        # lr_0*d^floor((1+k)/r)
        if self.smoothing:
            return 0.5*(0.01**np.floor((1+self.iteration)/len(self.nodes)))
        return 1*(0.02**np.floor((1+self.iteration)/len(self.nodes)))
    def findWinner(self,input):
        error=float("inf")
        winner = None
        for node in self.nodes:
            d = distance(input,node.array)
            if(d < error):
                error = d
                winner = node
        return winner, error
    def neighborhood(self,focus):
        radius = self.radius
        if self.smooth:
            radius = radius / 3
        neighborhood = []
        for node in self.nodes:
            if distance(node.array,focus.array) < radius:
                neighborhood.append(node)
        return neighborhood
    def basisInputNeighborhood(self,focus,radius):
    
        inputNeighborhood = []
        for inputData in self.data:
            if distance(inputData,focus.array) < radius:
                inputNeighborhood.append(inputData)
        return inputNeighborhood

    def averageError(self):
        error = 0
        for node in self.nodes:
            error += node.error
        return error / len(self.nodes)
    def maxError(self):
        error = 0
        for node in self.nodes:
            if node.error > error:
                error = node.error
        return error

    def createClosestBasis(self):
        new_basis = [self.basis[0]]
        unseen = self.basis[1:]
        last_added = self.basis[0]
        while len(new_basis) != len(self.basis):
            node_list = []
            for node in unseen:
                node_list.append((node,distance(node.array,last_added.array)))
            node_list = sorted(node_list,key=getKey)
            new_node = node_list[0][0]
            # print(unseen)
            # print(new_node)
            unseen.remove(new_node)
            new_basis.append(new_node)
            last_added = new_node

        self.basis = new_basis
    def connectNodes(self):
        origin = np.array([0,0])
        
        basis_nodes = []
        for node in self.nodes:
            basis_nodes.append((node,distance(node.array, origin)))
        basis_nodes = sorted(basis_nodes,key=getKey)
        
        self.basis = []
        for pair in basis_nodes:
            self.basis.append(pair[0])
        self.createClosestBasis()
        return self.basis
    
    def adaptWeights(self,winning_node,input):
        # Adapt weight of winner and nodes within neighborhood
        #   Weights are distributed proportionally to neighborhood nodes by distance to winner
        # N_k+1 is the neighborhood of winning neuron at k+1 iteration
        # LR(k) should depend on number of nodes at time k
        # w_j(k+1) = w_j(k)                          if j not in N_k+1
        #            w_j(k) + LR(k)*(x_k-w_j(k))     if j in N_k+1
        # where LR(k) is learning rate of K in N, sequence of positive parameters converging to 0 as k->infinity
        # w_j(k),w_j(k+1) are nodes, j, before and after adaption

        for node in self.neighborhood(winning_node):
            node.array = node.array + self.learningRate() * distance(node.array,input)
    def growNode(self,winning_node,input):
        if self.connected:
            if winning_node.left != None and winning_node.right != None:
                input_node = Node(input[0],input[1])
                new_node = middleNode(winning_node,input_node)
                if distance(winning_node.left.array, input_node.array) < distance(winning_node.right.array, input_node.array):
                    # Add new node to winning_nodes left
                    # winning_node.left --- new_node --- winning_node
                    winning_node.left.right = new_node
                    new_node.right = winning_node
                    new_node.left = winning_node.left
                    winning_node.left = new_node
                    
                    self.nodes.append(new_node)
                else: 
                    # winning_node --- new_node --- winning_node.right
                    winning_node.right.left = new_node
                    new_node.left = winning_node
                    winning_node.right = new_node
                    new_node.left = winning_node
                    self.nodes.append(new_node)
                # pass
            elif winning_node.left == None:
                new_node = Node(input[0],input[1])
                winning_node.left = new_node
                new_node.right = winning_node
                self.nodes.append(new_node)
            elif winning_node.right == None:
                new_node = Node(input[0],input[1])
                winning_node.right = new_node
                new_node.left = winning_node
                self.nodes.append(new_node)
        else: 
            new_node = Node(input[0],input[1])
            self.nodes.append(new_node)

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
                    print("grow")
                    self.growNode(winning_node,inputData)

                self.iteration += 1
            average_error = self.averageError()
            print(f"Average error {average_error}")            
            max_error = self.maxError()
            print(f"Max error {average_error}")

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
                            print("grow")
                            self.growNode(winning_node,inputData)

                        self.iteration += 1                  
                
                
            
        # Reset LR to start value


    def smoothing(self):
        # Reduce learning rate and decrease neighborhood
        # Find winners and adapt weights as in growing phase
        self.smooth = True
        for i in range(25):
           for inputData in self.data:
                # print(input)
                # print(f"Iteration #: {self.iteration}")

                winning_node, error = self.findWinner(inputData)
                self.adaptWeights(winning_node, inputData)

                # Increase error value of winner (difference between input and node)
                # When error value > growth threshold, grow node if winner is a boundary node
                # Otherwise  distribute weights

                winning_node.addError(error)

                if winning_node.error > self.growth_threshold:
                    print("grow")
                    self.growNode(winning_node, inputData)

                self.iteration += 1 
                # average_error = self.averageError()
                # print(f"Average error {average_error}")
                # max_error = self.maxError()
                # print(f"Max error {average_error}")



    def train(self):
        if self.connected:
            self.growingConnected()
        else:
            self.growing()
        self.smoothing()

    def print(self):
        if self.connected:
            node = self.nodes[0]
            while node.left != None:
                node = node.left
            nodes = [node]
            while node.right != None:
                nodes.append(node.right)
                node = node.right
            return nodes
        else:
            return self.nodes