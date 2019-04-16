import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import scipy.linalg



def ClusterGraph(Clusters,ClusterNodes,ClusterConnections,ClusterDist=20):
    '''generate random graph of unconnected clusters, each with same fixed number of nodes and edges'''

    TotNodes = Clusters*ClusterNodes

    G = nx.Graph(directed=False)

    NodeList = []
    for i in range(TotNodes):
        NodeList.append(i)

    G.add_nodes_from(NodeList)
    pos = dict()

    for cluster in range(Clusters):
        for _ in range(ClusterConnections):
            node1 = np.random.randint(ClusterNodes*cluster, ClusterNodes+cluster*ClusterNodes)
            node2 = np.random.randint(ClusterNodes*cluster, ClusterNodes+cluster*ClusterNodes)
            G.add_edge(node1, node2)

        X_G = np.random.uniform(float(cluster*ClusterDist),float(cluster*ClusterDist+10),ClusterNodes)
        Y_G = np.random.uniform(0.0,15.0,ClusterNodes)

        for j in range(ClusterNodes):
            pos[j+(cluster*ClusterNodes)] = (X_G[j], Y_G[j])
    
    return G, pos


def ConnectRand(G, ConnTot, Clusters):

    TotNodes = G.number_of_nodes()
    ClusterNodes = G.number_of_nodes()/Clusters

    for _ in range(ConnTot):
        InterNode1 = np.random.randint(0, TotNodes)
        InterNode2 = np.random.randint(0, TotNodes)
        while(int(InterNode1/ClusterNodes) == int(InterNode2/ClusterNodes)):
            InterNode2 = np.random.randint(0, TotNodes)

        G.add_edge(InterNode1,InterNode2)
    
    return G

def ConnectNext(G, NextConns, Clusters):

    TotNodes = G.number_of_nodes()
    ClusterNodes = TotNodes/Clusters
    
    for C in range(Clusters-1):
        for _ in range(NextConns):
            NextNode1 = np.random.randint(C*ClusterNodes, (C+1)*ClusterNodes)
            NextNode2 = np.random.randint( (C+1)*ClusterNodes,  (C+2)*ClusterNodes)

            G.add_edge(NextNode1,NextNode2)

    return G




trials = 5
AvgTrial = 5
steps = 200
gamma = 1.0
eta = 1.0
StepsToHalfRand = np.zeros(trials)
StepsToHalfNext = np.zeros(trials)
ArrPnext = np.zeros((trials,steps))
ArrPrandom = np.zeros((trials,steps))

for t in range(2):
    if t == 0:
        conntype = 'next'
    elif t == 1:
        conntype = 'random'

    for _ in tqdm(range(AvgTrial)):         
        Clusters = 10 # 3 # 10
        ClusterNodes = 10 # 18 # 10
        for j in tqdm(range(trials)):

            G, pos = ClusterGraph(Clusters, ClusterNodes, int(((5*ClusterNodes - 10)/4)+1)) # same number of cluster edges as in one column of hexagons

            if conntype == 'random':
                G = ConnectRand(G, int((((ClusterNodes-2)/4)+1)*(Clusters-1)), Clusters) # same total number of edges as 'next'
            elif conntype == 'next':
                G = ConnectNext(G, int(((ClusterNodes-2)/4)+1), Clusters) # same number of inter-cluster edges as one column in hexagons
        
            TotNodes = G.number_of_nodes()
        
            Adj = nx.adjacency_matrix(G)
            Adj = Adj.todense()

            Deg = np.zeros((TotNodes,TotNodes))
            for i in range(TotNodes):
                Deg[i,i] = np.sum(Adj[i,:])

            H = np.zeros((TotNodes,TotNodes),dtype=complex)
            H += gamma*(Deg-Adj)

            for key, val in pos.items():
                if (val[0] >= (Clusters)*20-20 and val[0] <= (Clusters)*20):
                    H[key,key] -= 1j*(eta/2)


            for step in tqdm(range(steps)):
                U = scipy.linalg.expm(-1j*H*step)
                psi0 = np.zeros(TotNodes, dtype=complex)
                for key, val in pos.items():
                    if (val[0] >= 0 and val[0] <= 10): # particle starts at the left
                        psi0[key] = 1 # superposition of all nodes on the left
                psi0 = psi0/(np.sqrt(sum(psi0)))
                psiN = np.dot(U,psi0)

                probs = abs(psiN**2)

                ArrP = 1.0-sum(probs)
                if conntype == 'next':
                    ArrPnext[j][step] += ArrP 
                if conntype == 'random':
                    ArrPrandom[j][step] += ArrP 

                if abs(ArrP - 0.5) < 0.05:
                    if conntype == 'random' and StepsToHalfRand[j]==0:
                        StepsToHalfRand[j] += step
                    elif conntype == 'next' and StepsToHalfNext[j]==0:
                        StepsToHalfNext[j] += step
                    # break

            ClusterNodes += 4


StepsToHalfNext = StepsToHalfNext/AvgTrial
StepsToHalfRand = StepsToHalfRand/AvgTrial

ArrPnext = ArrPnext/AvgTrial
ArrPrandom = ArrPrandom/AvgTrial

print(StepsToHalfNext)
print(StepsToHalfRand)

fig = plt.figure(figsize=(14,14), dpi=200)

ax1 = fig.add_subplot(211) # random conns
for i in range(trials):
    plt.plot(np.arange(steps), ArrPrandom[i], label=(str(6+(i+1)*4)+' nodes')) # label=(str(6+(i+1)*4)+' nodes')) # label=(str((i+1)*3)+' clusters'))
plt.xlabel('steps', fontsize=24)
plt.ylabel('arrival probability', fontsize=24)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.0), fontsize=24)
ax1.tick_params(labelsize=24)

ax2 = fig.add_subplot(212) # next-cluster conns

for i in range(trials):
    plt.plot(np.arange(steps), ArrPnext[i], label=(str(6+(i+1)*4)+' nodes')) # label=(str(6+(i+1)*4)+' nodes')) # label=(str((i+1)*3)+' clusters'))
plt.xlabel('steps', fontsize=24)
plt.ylabel('arrival probability', fontsize=24)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.0), fontsize=24)
ax2.tick_params(labelsize=24)


# ax3 = fig.add_subplot(313)

# xAx = np.arange(trials)
# xAx = 6+(xAx+1)*4

# plt.plot(xAx, StepsToHalfNext, label='next-cluster connections')
# plt.plot(xAx, StepsToHalfRand, label='random connections')
# plt.xlabel('number of nodes')
# plt.ylabel('steps')

# plt.legend(loc='upper left', bbox_to_anchor=(1, 1.0))

plt.subplots_adjust(right=0.70)
plt.show()