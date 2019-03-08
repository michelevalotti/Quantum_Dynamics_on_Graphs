import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import scipy.linalg
from ClusterArrProb import ClusterGraph, ConnectNext, ConnectRand
from ClusterSpreading import StandardDeviation

'''to compare to hexagons each cluster should have same number of nodes as one row of hexagons
(perpendicular to direction of travel), and the number of clusters should be the same as the
number of hexagons in the directions of travel. Tune the inter-cluster edges to get similar
standard dev'''


def ConnectAllNext(G, NextConns, Clusters):

    ClusterNodes = G.number_of_nodes()/Clusters

    for C in range(Clusters-1):
        for CNext in range(NextConns):
            NextNode1 = np.random.randint(C*ClusterNodes, (C+1)*ClusterNodes)
            NextNode2 = np.random.randint((C+1)*ClusterNodes, (C+2)*ClusterNodes)
            while(int(NextNode1/ClusterNodes) == int(NextNode2/ClusterNodes)):
                NextNode2 =  np.random.randint((C+1)*ClusterNodes, (C+2)*ClusterNodes)

            G.add_edge(NextNode1,NextNode2)

    return G


MyG = ClusterGraph(4,3,6)[0]
MyPos = ClusterGraph(4,3,6)[1]

MyG = ConnectNext(MyG, 4, 4)

nx.draw_networkx_nodes(MyG, MyPos)
nx.draw_networkx_edges(MyG, MyPos)
plt.show()