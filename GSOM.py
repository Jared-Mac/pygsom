from platform import node
import numpy as np
from sklearn.metrics import euclidean_distances

def distance(n1,n2):
    return np.linalg.norm(n1-n2)
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
        n3 = Node(self.data[-1][0],self.data[-1][1])
        n1.right = n2
        n2.left, n2.right = n1,n3
        n3.left = n2
        self.nodes.append(n1)
        self.nodes.append(n2)
        self.nodes.append(n3)

    def __init__(self,dataset,spread_factor,radius):
        # Possible parameters
        # Neighborhood function (Gauss vs Bubble vs Combined)
        # Neighborhood Size
        # Terminal Condition (Such as preset iteration number or low node growth level)
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

    def learningRate(self):
        # learning rate = at iteration k, with r nodes, initial learning rate lr_0
        # d is constant, how much the learning rate should drop
        # lr_0*d^floor((1+k)/r)
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

        neighborhood = []
        for node in self.nodes:
            if distance(node.array,focus.array) < self.radius:
                neighborhood.append(node)
        return neighborhood

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
        # if winning_node.left != None and winning_node.right != None:
        #     pass
        # if winning_node.left == None:
        #     new_node = Node(input[0],input[1])
        #     winning_node.left = new_node
        #     self.nodes.append(new_node)
        # if winning_node.right == None:
        #     new_node = Node(input[0],input[1])
        #     winning_node.right = new_node
        #     self.nodes.append(new_node)
        new_node = Node(input[0],input[1])
        self.nodes.append(new_node)

        winning_node.error = 0

    def growing(self):
        # Iterate until terminal condition
        # Grab data point
        for x in range(50):
            for input in self.data:
                # print(input)
                # print(f"Iteration #: {self.iteration}")
                winning_node,error = self.findWinner(input)
                self.adaptWeights(winning_node,input)


                # Increase error value of winner (difference between input and node)
                # When error value > growth threshold, grow node if winner is a boundary node
                # Otherwise  distribute weights

                winning_node.addError(error)

                if winning_node.error > self.growth_threshold:
                    print("grow")
                    self.growNode(winning_node,input)

                self.iteration += 1
            
            
            
        # Reset LR to start value



    def smoothing(self):
        # Reduce learning rate and decrease neighborhood
        # Find winners and adapt weights as in growing phase
        pass

    def train(self):
        self.growing()
        self.smoothing()

    def print(self):
        for node in self.nodes:
            print(node.array)