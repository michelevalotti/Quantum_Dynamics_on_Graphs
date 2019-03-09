import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import scipy.linalg
from ClusterArrProb import ClusterGraph, ConnectRand, ConnectNext


def StandardDeviation(G, ClusterLen, pos, steps, gamma=1.0):

    TotNodes = G.number_of_nodes()

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

        MiddleX = (int((abs((pos[0])[0] - (pos[TotNodes-1])[0])/2)))


        for key, val in pos.items():
            xPosns[key] = val[0]
            # if (val[0] >= 40 and val[0] <= 50): # particle starts in the middle
            if (val[0] >= (MiddleX) and val[0] <= (MiddleX+ClusterLen)): # particle starts in the middle
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
    
    TrialsTot = 30 # avegrage over this many trials

    # fixed variables
    Clusters = 5
    ClusterLen = 10 # depends on graph making fn, specify for StdDev
    if MyVar == 'Nodes':
        NextConns_low = 15
        NextConns_high = 15
    if MyVar == 'Edges':
        ClusterNodes_low = 10
        ClusterNodes_high = 10
        ClusterConnections = 20
    gamma = 1.0
    steps = 30


    StandDevArr_next_low_avg = np.zeros(steps)
    StandDevArr_rand_low_avg = np.zeros(steps)
    StandDevArr_next_high_avg = np.zeros(steps)
    StandDevArr_rand_high_avg = np.zeros(steps)

            
    fig = plt.figure()


    for trial in tqdm(range(TrialsTot)):

        # low nodes/edges graph
        if MyVar == 'Nodes':
            ClusterNodes_low = 10
            ClusterConnections = 20
        if MyVar == 'Edges':
            NextConns_low = 15

        G = ClusterGraph(Clusters, ClusterNodes_low, ClusterConnections)[0]
        pos = ClusterGraph(Clusters, ClusterNodes_low, ClusterConnections)[1]
        TotNodes = ClusterGraph(Clusters, ClusterNodes_low, ClusterConnections)[2]

        # next conns, low nodes/edges
        G_next = ConnectNext(G, NextConns_low, Clusters)
        StandDevArr_next_low = StandardDeviation(G_next, ClusterLen, pos, steps, gamma)[0]
        StandDevArr_next_low_avg += StandDevArr_next_low
        probs_next = StandardDeviation(G_next, ClusterLen, pos, steps, gamma)[1]

        # rand conns, low nodes/edges
        G_rand = ConnectRand(G, NextConns_low, Clusters)
        StandDevArr_rand_low = StandardDeviation(G_rand, ClusterLen, pos, steps, gamma)[0]
        StandDevArr_rand_low_avg += StandDevArr_rand_low
        probs_rand = StandardDeviation(G_rand, ClusterLen, pos, steps, gamma)[1]

        if trial == 0:
            ax1 = fig.add_subplot(311)
            node_size = probs_next*(100000/(max(probs_next)*TotNodes)) # rescale so size of node is never too small or too big
            PltNodes = nx.draw_networkx_nodes(G_next, pos, node_color=probs_next, node_size=node_size)
            PltEdges = nx.draw_networkx_edges(G_next, pos)
            col1 = fig.colorbar(PltNodes, label='Probability', shrink=0.9)


        # high nodes/edges graph
        if MyVar == 'Nodes':
            ClusterNodes_high = 40
            ClusterConnections = 80
        if MyVar == 'Edges':
            NextConns_high = 60

        G = ClusterGraph(Clusters, ClusterNodes_high, ClusterConnections)[0]
        pos = ClusterGraph(Clusters, ClusterNodes_high, ClusterConnections)[1]
        TotNodes = ClusterGraph(Clusters, ClusterNodes_high, ClusterConnections)[2]

        # next conns, low nodes/edges
        G_next = ConnectNext(G, NextConns_high, Clusters)
        StandDevArr_next_high = StandardDeviation(G_next, ClusterLen, pos, steps, gamma)[0]
        StandDevArr_next_high_avg += StandDevArr_next_high
        probs_next = StandardDeviation(G_next, ClusterLen, pos, steps, gamma)[1]

        # rand conns, low nodes/edges
        G_rand = ConnectRand(G, NextConns_high, Clusters)
        StandDevArr_rand_high = StandardDeviation(G_rand, ClusterLen, pos, steps, gamma)[0]
        StandDevArr_rand_high_avg += StandDevArr_rand_high
        probs_rand = StandardDeviation(G_rand, ClusterLen, pos, steps, gamma)[1]

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

    if MyVar == 'Nodes':
        LabelVar_low = ClusterNodes_low
        LabelVar_high = ClusterNodes_high
    if MyVar == 'Edges':
        LabelVar_low = NextConns_low
        LabelVar_high = NextConns_high

    plt.plot(np.arange(steps), StandDevArr_next_low_avg, label=('next-cluster connections - ' + str(LabelVar_low) + ' ' + MyVar))
    plt.plot(np.arange(steps), StandDevArr_rand_low_avg, label=('random connections - ' + str(LabelVar_low) + ' ' + MyVar))
    plt.plot(np.arange(steps), StandDevArr_next_high_avg, label=('next-cluster connections - ' + str(LabelVar_high) + ' ' + MyVar))
    plt.plot(np.arange(steps), StandDevArr_rand_high_avg, label=('random connections - ' + str(LabelVar_high) + ' ' + MyVar))
    plt.xlabel('steps')
    plt.ylabel('$\sigma_x$ from starting posn')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.7)

    plt.show()