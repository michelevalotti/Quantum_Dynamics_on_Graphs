import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import scipy.linalg


def ClusterGraph(Clusters,ClusterNodes,ClusterConnections):
    '''generate random graph of unconnected clusters, each with same fixed number of nodes and edges'''

    TotNodes = Clusters*ClusterNodes

    G = nx.Graph(directed=False)

    NodeList = []
    for i in range(TotNodes):
        NodeList.append(i)

    G.add_nodes_from(NodeList)
    pos = dict()

    for cluster in range(Clusters):
        for c in range(ClusterConnections):
            node1 = np.random.randint(ClusterNodes*cluster, ClusterNodes+cluster*ClusterNodes)
            node2 = np.random.randint(ClusterNodes*cluster, ClusterNodes+cluster*ClusterNodes)
            G.add_edge(node1, node2)

        X_G = np.random.uniform(float(cluster*20),float(cluster*20+10),ClusterNodes)
        Y_G = np.random.uniform(0.0,15.0,ClusterNodes)

        for j in range(ClusterNodes):
            pos[j+(cluster*ClusterNodes)] = (X_G[j], Y_G[j])
    
    return G, pos, TotNodes



def ClusterArrivalProb(G,ConnTrials,steps,InterClusterConn,gamma,eta):
    '''add edges between clusters and record probability that quantum walker reacher cluster on the opposite side'''

    ArrivalProbs = []
    TotInterConn = []

    for InterConn in range(InterClusterConn):
        InterNode1 = np.random.randint(0, TotNodes)
        InterNode2 = np.random.randint(0, TotNodes)
        while(int(InterNode1/ClusterNodes) == int(InterNode2/ClusterNodes)):
            InterNode2 = np.random.randint(0, TotNodes)
        
        G.add_edge(InterNode1,InterNode2)

    for j in tqdm(range(ConnTrials)):

        InterNode1 = np.random.randint(0, TotNodes)
        InterNode2 = np.random.randint(0, TotNodes)
        while(int(InterNode1/ClusterNodes) == int(InterNode2/ClusterNodes)):
            InterNode2 = np.random.randint(0, TotNodes)

        G.add_edge(InterNode1,InterNode2)


        if j == 0:
            G_init = G.copy()

        Adj = nx.adjacency_matrix(G)
        Adj = Adj.todense()

        Deg = np.zeros((TotNodes,TotNodes))
        for i in range(TotNodes):
            Deg[i,i] = np.sum(Adj[i,:])

        H = np.zeros((TotNodes,TotNodes),dtype=complex)
        H += gamma*(Deg-Adj)

        for key, val in pos.items():
            if (val[0] >= Clusters*20-20 and val[0] <= Clusters*20):
                H[key,key] -= 1j*(eta/2)


        for step in (range(steps)):
            U = scipy.linalg.expm(-1j*H*step)
            psi0 = np.zeros(TotNodes, dtype=complex)
            for key, val in pos.items():
                if (val[0] >= 0 and val[0] <= 10): # particle starts at the left
                    psi0[key] = 1 # superposition of all nodes on the left
            psi0 = psi0/(np.sqrt(sum(psi0)))
            psiN = np.dot(U,psi0)

            probs = abs(psiN**2)

        if j == 0:
            probs_init = probs.copy()

        ArrivalProbs.append(1.0-sum(probs))
        InterClusterConn += 1
        TotInterConn.append(InterClusterConn)

    return G_init, G, probs_init, probs, ArrivalProbs, TotInterConn


Clusters = 2 # initial number of clusters
ClustersLegend = Clusters # for plotting, value of 'Clusters' changes
ClusterNodes = 20 # number of nodes in each cluster
ClusterConnections = 40 # number of edges in each cluster
InterClusterConn = 5 # initial number of edges between clusters
gamma = 1.0
eta = 1.0
steps = 4*Clusters # steps taken by quantum walker
ConnTrials = 100 # each trial adds an inter-cluster edge
clusterTrials = 6 # each trial adds a cluster


ArrivalProbsAll = []
MyGraphsAll = []
MyProbsAll = []
MyPosAll = []
MyTotNodesAll = []


for c in range(clusterTrials):
    '''calculate arrival probability wrt number of connections between clusters, for graphs with increasing number of clusters'''

    print(Clusters, ' clusters, ', 'total number of nodes: ', Clusters*ClusterNodes)

    MyGraph = ClusterGraph(Clusters,ClusterNodes,ClusterConnections) # generate graph (no inter cluster connections)
    G = MyGraph[0]
    pos = MyGraph[1]
    TotNodes = MyGraph[2]

    MyClusterProb = ClusterArrivalProb(G,ConnTrials,steps,InterClusterConn,gamma,eta) # add inter cluster connections and measure arrival prob

    G_init = MyClusterProb[0]
    G = MyClusterProb[1]
    probs_init = MyClusterProb[2]
    probs = MyClusterProb[3]
    ArrivalProbs = MyClusterProb[4]
    TotInterConn = MyClusterProb[5]

    MyGraphsAll.append((G_init,G))
    MyProbsAll.append((probs_init, probs))
    ArrivalProbsAll.append(ArrivalProbs)
    MyPosAll.append(pos)
    MyTotNodesAll.append(TotNodes)

    print('Arrival Prob: ', ArrivalProbs[ConnTrials-1])

    Clusters += 1


# plot

fig = plt.figure()

ax1 = fig.add_subplot(311)

node_size_init = MyProbsAll[clusterTrials-1][0]*(100000/(max(MyProbsAll[clusterTrials-1][0])*MyTotNodesAll[clusterTrials-1])) # rescale so size of node is never too small or too big
PltNodes = nx.draw_networkx_nodes(MyGraphsAll[clusterTrials-1][0],MyPosAll[clusterTrials-1], node_color=MyProbsAll[clusterTrials-1][0], node_size=node_size_init)
pltEdges = nx.draw_networkx_edges(MyGraphsAll[clusterTrials-1][0],MyPosAll[clusterTrials-1])
col1 = fig.colorbar(PltNodes, label='Probability', shrink=0.9)

ax2 = fig.add_subplot(312)

node_size = MyProbsAll[clusterTrials-1][1]*(100000/(max(MyProbsAll[clusterTrials-1][1])*MyTotNodesAll[clusterTrials-1])) # rescale so size of node is never too small or too big
PltNodes = nx.draw_networkx_nodes(MyGraphsAll[clusterTrials-1][1],MyPosAll[clusterTrials-1], node_color=MyProbsAll[clusterTrials-1][1], node_size=node_size)
PltEdges = nx.draw_networkx_edges(MyGraphsAll[clusterTrials-1][1],MyPosAll[clusterTrials-1])
col2 = fig.colorbar(PltNodes, label='Probability', shrink=0.9)

ax3 = fig.add_subplot(313)

for p in range(clusterTrials):
    plt.plot(TotInterConn,ArrivalProbsAll[p], label=(str(ClustersLegend+p)+' clusters'))
plt.ylabel('Arrival Probability')
plt.xlabel('Number of inter-cluster edges')
plt.legend()

plt.show()