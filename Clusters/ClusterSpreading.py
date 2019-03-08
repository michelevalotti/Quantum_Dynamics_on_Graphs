import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import scipy.linalg
from ClusterArrProb import ClusterGraph, ConnectRand, ConnectNext


def StandardDeviation(G, steps, gamma):

    Adj = nx.adjacency_matrix(G)
    Adj = Adj.todense()

    Deg = np.zeros((TotNodes,TotNodes))
    for i in range(TotNodes):
        Deg[i,i] = np.sum(Adj[i,:])

    H = np.zeros((TotNodes,TotNodes),dtype=complex)
    H += gamma*(Deg-Adj)

    StandDevArr = np.zeros(steps)

    for step in (range(steps)):
        U = scipy.linalg.expm(-1j*H*step)
        psi0 = np.zeros(TotNodes, dtype=complex)
        xPosns = np.zeros(TotNodes)
        AvgStart = 0.0
        for key, val in pos.items():
            xPosns[key] = val[0]
            if (val[0] >= 40 and val[0] <= 50): # particle starts in the middle
                psi0[key] = 1 # superposition of all nodes in the middle
                AvgStart += float(val[0])
        AvgStart = AvgStart/abs(sum(psi0))
        xPosns -= AvgStart
        psi0 = psi0/(np.sqrt(sum(psi0)))
        psiN = np.dot(U,psi0)

        probs = abs(psiN**2)

        AvgX = sum(xPosns*probs)
        AvgXsq = sum((xPosns**2)*probs)

        StandDev = np.sqrt((AvgXsq - (AvgX)**2))
        StandDevArr[step] = StandDev

    return StandDevArr, probs


if __name__ =='__main__':

    '''fixed nodes and varying edges (random or next)
    or fixed edges (random or next) and varying nodes'''

    MyVar = 'Nodes' # 'Nodes' or 'Edges' -- nodes for variable number of nodes, edges for variable number of edges
    TrialsTot = 10 # avegrage over this many trials

    '''high and low number of nodes for next conns, and high and low mumber of nodes for random conns'''

    # fixed variables
    Clusters = 5
    if MyVar == 'Nodes':
        NextConns = 15
    if MyVar == 'Edges':
        ClusterNodes = 10
        ClusterConnections = 20
    gamma = 1.0
    steps = 20


    StandDevArr_next_low_avg = np.zeros(steps)
    StandDevArr_rand_low_avg = np.zeros(steps)
    StandDevArr_next_high_avg = np.zeros(steps)
    StandDevArr_rand_high_avg = np.zeros(steps)

            
    fig = plt.figure()


    for trial in tqdm(range(TrialsTot)):

        # low nodes/edges graph
        if MyVar == 'Nodes':
            ClusterNodes = 10
            ClusterConnections = 20
        if MyVar == 'Edges':
            NextConns = 15

        G = ClusterGraph(Clusters, ClusterNodes, ClusterConnections)[0]
        pos = ClusterGraph(Clusters, ClusterNodes, ClusterConnections)[1]
        TotNodes = ClusterGraph(Clusters, ClusterNodes, ClusterConnections)[2]

        # next conns, low nodes/edges
        G_next = ConnectNext(G, NextConns, Clusters)
        StandDevArr_next_low = StandardDeviation(G_next, steps, gamma)[0]
        StandDevArr_next_low_avg += StandDevArr_next_low
        probs_next = StandardDeviation(G_next, steps, gamma)[1]

        # rand conns, low nodes/edges
        G_rand = ConnectRand(G, NextConns, Clusters)
        StandDevArr_rand_low = StandardDeviation(G_rand, steps, gamma)[0]
        StandDevArr_rand_low_avg += StandDevArr_rand_low
        probs_rand = StandardDeviation(G_rand, steps, gamma)[1]

        if trial == 0:
            ax1 = fig.add_subplot(311)
            node_size = probs_next*(100000/(max(probs_next)*TotNodes)) # rescale so size of node is never too small or too big
            PltNodes = nx.draw_networkx_nodes(G_next, pos, node_color=probs_next, node_size=node_size)
            PltEdges = nx.draw_networkx_edges(G_next, pos)
            col1 = fig.colorbar(PltNodes, label='Probability', shrink=0.9)


        # high nodes/edges graph
        if MyVar == 'Nodes':
            ClusterNodes = 40
            ClusterConnections = 80
        if MyVar == 'Edges':
            NextConns = 60

        G = ClusterGraph(Clusters, ClusterNodes, ClusterConnections)[0]
        pos = ClusterGraph(Clusters, ClusterNodes, ClusterConnections)[1]
        TotNodes = ClusterGraph(Clusters, ClusterNodes, ClusterConnections)[2]

        # next conns, low nodes/edges
        G_next = ConnectNext(G, NextConns, Clusters)
        StandDevArr_next_high = StandardDeviation(G_next, steps, gamma)[0]
        StandDevArr_next_high_avg += StandDevArr_next_high
        probs_next = StandardDeviation(G_next, steps, gamma)[1]

        # rand conns, low nodes/edges
        G_rand = ConnectRand(G, NextConns, Clusters)
        StandDevArr_rand_high = StandardDeviation(G_rand, steps, gamma)[0]
        StandDevArr_rand_high_avg += StandDevArr_rand_high
        probs_rand = StandardDeviation(G_rand, steps, gamma)[1]

        if trial == 0:
            ax2 = fig.add_subplot(312)
            node_size = probs_rand*(100000/(max(probs_rand)*TotNodes)) # rescale so size of node is never too small or too big
            PltNodes = nx.draw_networkx_nodes(G_rand, pos, node_color=probs_rand, node_size=node_size)
            PltEdges = nx.draw_networkx_edges(G_rand, pos)
            col1 = fig.colorbar(PltNodes, label='Probability', shrink=0.9)


    StandDevArr_next_low_avg = StandDevArr_next_low_avg / TrialsTot
    StandDevArr_rand_low_avg = StandDevArr_rand_low_avg / TrialsTot
    StandDevArr_next_high_avg = StandDevArr_next_high_avg / TrialsTot
    StandDevArr_rand_high_avg = StandDevArr_rand_high_avg / TrialsTot
    
    
    ax3 = fig.add_subplot(313)
    plt.plot(np.arange(steps), StandDevArr_next_low_avg, label='next-cluster connections - low')
    plt.plot(np.arange(steps), StandDevArr_rand_low_avg, label='random connections - low')
    plt.plot(np.arange(steps), StandDevArr_next_high_avg, label='next-cluster connections - high')
    plt.plot(np.arange(steps), StandDevArr_rand_high_avg, label='random connections - high')
    plt.xlabel('steps')
    plt.ylabel('$\sigma_x$ from starting posn')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.7)

    plt.show()