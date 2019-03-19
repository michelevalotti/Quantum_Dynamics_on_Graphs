import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from tqdm import tqdm
import scipy.linalg
from ClusterArrProb import ConnectRand
from ClusterSpreading import StandardDeviation


'''to compare to hexagons each cluster should have same number of nodes as one row of hexagons
(perpendicular to direction of travel), and the number of clusters should be the same as the
number of hexagons in the directions of travel. Tune the inter-cluster edges to get similar
standard dev'''



def HexCluster(Clusters,ClusterX,ClusterY,orientation):
    '''generate random graph of unconnected clusters, each with same fixed number of nodes and edges'''

    ClusterList = []
    PosDictList = []

    for cluster in range(Clusters):
        G = nx.hexagonal_lattice_graph(ClusterY,ClusterX)
        coords = []

        if orientation == 'horizontal':
            Cdist = ClusterX + 1 # distance between clusters, for drawing
            for i,j in G.nodes():
                coords.append((i+((ClusterX+1)*cluster),j)) # clusters are 1 hexagon apart
        if orientation == 'vertical':
            Cdist = ClusterY*2+2
            for i,j in G.nodes():
                coords.append((i,j+((ClusterY*2+2)*cluster)))
        
        
        # shift positions to make graph look like a hexagonal lattice
        for coord in range(len(coords)):
            if (((coords[coord][0]-(Cdist*cluster))%2 == 0) and (coords[coord][1]%2 != 0)):
                coords[coord] = ((float(coords[coord][0]) - 0.15), coords[coord][1])
            elif (((coords[coord][0]-(Cdist*cluster))%2 != 0) and (coords[coord][1]%2 != 0)):
                coords[coord] = ((float(coords[coord][0]) + 0.15), coords[coord][1])
            elif (((coords[coord][0]-(Cdist*cluster))%2 == 0) and (coords[coord][1]%2 == 0)):
                coords[coord] = ((float(coords[coord][0]) + 0.15), coords[coord][1])
            elif (((coords[coord][0]-(Cdist*cluster))%2 != 0) and (coords[coord][1]%2 == 0)):
                coords[coord] = ((float(coords[coord][0]) - 0.15), coords[coord][1])


        NewLabels = [n for n in range(G.number_of_nodes())]
        LabelDict = dict(zip(G.nodes(),(np.array(NewLabels)+(G.number_of_nodes())*cluster))) # rename nodes from 0 to Clusters*ClusterNodes

        G = nx.relabel_nodes(G,LabelDict)
        ClusterList.append(G)

        PosDict = dict(zip((np.array(NewLabels)+(G.number_of_nodes())*cluster),coords))

        PosDictList.append(PosDict)
    

    PosDictAll = {}

    for c in range(0,Clusters):
        PosDictAll.update(PosDictList[c])

    G = ClusterList[0]
    for i in range(Clusters-1):
        G = nx.union(G,ClusterList[i+1])
    
    return G, PosDictAll



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

def ConnectHexNext(G, HexY, NextConns, Clusters, orientation):

    ClusterNodes = G.number_of_nodes()/Clusters

    if orientation == 'horizontal':
        for C in range(Clusters-1):
            for CNext in range(NextConns):
                NextNode1 = np.random.randint( (C*ClusterNodes)+(ClusterNodes-(2*HexY+1)), (C+1)*ClusterNodes)
                NextNode2 = np.random.randint((C+1)*ClusterNodes, ((C+1)*ClusterNodes)+(2*HexY+1) )
                G.add_edge(NextNode1,NextNode2)

    if orientation == 'vertical':
        for C in range(Clusters-1):
            TopNodes = []
            BottomNodes = []
            for k,v in pos.items():
                if v[1] == (C*(2*HexY+2))+(2*HexY+1) or v[1] == (C*(2*HexY+2))+(2*HexY):
                    TopNodes.append(k)
                if v[1] == ((C+1)*(2*HexY+2)+1) or v[1] == (C+1)*(2*HexY+2):
                    BottomNodes.append(k)
            for CNext in range(NextConns):
                NextNode1 = random.choice(TopNodes)
                NextNode2 = random.choice(BottomNodes)
                G.add_edge(NextNode1,NextNode2)

    return G


def HexClusterArr(G,ClusterY,pos,steps,orientation,gamma=1.0,eta=1.0):

    TotNodes = G.number_of_nodes()

    Adj = nx.adjacency_matrix(G)
    Adj = Adj.todense()

    Deg = np.zeros((TotNodes,TotNodes))
    for i in range(TotNodes):
        Deg[i,i] = np.sum(Adj[i,:])

    H = np.zeros((TotNodes,TotNodes),dtype=complex)
    H += gamma*(Deg-Adj)

    if orientation == 'horizontal':
        for l in range(TotNodes):
            if (l >= (TotNodes-(ClusterY*2 + 1)) and l <= TotNodes):
                H[l,l] -= 1j*(eta/2)

    if orientation == 'vertical':
        for k,v in pos.items():
            MaxY = v[1]
            if v[1] > MaxY:
                v[1] = MaxY
        for key,val in pos.items():
            if val[1] == MaxY or val[1] == (MaxY-1):
                H[key,key] -= 1j*(eta/2)


    EndProb = np.zeros(steps)

    for step in tqdm(range(steps)):
        U = scipy.linalg.expm(-1j*H*step)
        psi0 = np.zeros(TotNodes, dtype=complex)
        for l in range(TotNodes):
            if (l >= 0 and l <= (ClusterY*2)): # particle starts at the left
                psi0[l] = 1 # superposition of all nodes on the left
        psi0 = psi0/(np.sqrt(sum(psi0)))
        psiN = np.dot(U,psi0)

        probs = abs(psiN**2)

        EndProb[step] = 1 - sum(probs)
    
    return EndProb, probs



if __name__ == '__main__':

    # plot

    fig = plt.figure()

    orientation = 'horizontal' # 'vertical' or 'horizontal'
    Clusters = 10 # keep higher than 1
    ClusterX = 2
    ClusterY = 2
    if orientation == 'horizontal':
        NextConns = (ClusterY*2)+1
        SDlabel = '$\sigma_x$'
    if orientation == 'vertical':
        NextConns = (ClusterX*2)+2
        SDlabel = '$\sigma_y$'
    steps = 10
    stepsArrProb = steps*4
    TotTrials = 1

    MyHexC = HexCluster(Clusters,ClusterX,ClusterY,orientation)

    G = MyHexC[0]
    pos = MyHexC[1]


    G_Rand = ConnectRand(G.copy(),NextConns*Clusters, Clusters)
    G_AllNext = ConnectAllNext(G.copy(), NextConns, Clusters)
    G_HexNext = ConnectHexNext(G.copy(), ClusterY, NextConns, Clusters, orientation)

    GList = [G_Rand,G_AllNext,G_HexNext]
    GListLabel = ['Random Connections', 'Next-Cluster Connections', 'Cluster-Edge Connections']

    # use gs for plotting
    gs1 = gridspec.GridSpec(5, 3)
    SubPlotList = [511,512,513,514,515]

    ArrProblist = np.zeros((3,stepsArrProb))

    for i in range(3):
        for j in range(TotTrials):
            print('trial ', j)
            MyArrP =  HexClusterArr(GList[i],ClusterY,pos,stepsArrProb,orientation)
            ArrP = MyArrP[0]
            ArrProblist[i] += ArrP
            probs = MyArrP[1]

        # ax1 = fig.add_subplot(SubPlotList[i])
        if orientation == 'horizontal':
            ax1 = fig.add_subplot(gs1[i,:])
        if orientation == 'vertical':
            ax1 = fig.add_subplot(gs1[:3,i])
        node_size = probs*(100000/(max(probs)*GList[i].number_of_nodes()))
        PltNodes = nx.draw_networkx_nodes(GList[i],pos, node_color=probs, node_size=node_size)
        PltEdges = nx.draw_networkx_edges(GList[i],pos)
        col1 = fig.colorbar(PltNodes, label='Probability', shrink=0.9)

    ArrProblist = ArrProblist/TotTrials

    SDevList = np.zeros((3,steps))

    for i in tqdm(range(3)):
        for j in range(TotTrials):
            print('trial ', j)
            if orientation == 'horizontal':
                ClusterLen = ClusterX
            if orientation == 'vertical':
                ClusterLen = ClusterY
            MySDev = StandardDeviation(GList[i],nx.adjacency_matrix(GList[i]).todense(),ClusterLen,pos,steps,orientation=orientation)
            SDev = MySDev[0]
            SDevList[i] += SDev
    
    SDevList = SDevList/TotTrials


    # ax2 = fig.add_subplot(514)
    ax2 = fig.add_subplot(gs1[3,:])

    for d in range(3):
        plt.plot(np.arange(steps),SDevList[d], label=GListLabel[d])
    plt.xlabel('steps')
    plt.ylabel(SDlabel)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.7)

    # ax3 = fig.add_subplot(515)
    ax3 = fig.add_subplot(gs1[4,:])

    for a in range(3):
        plt.plot(np.arange(stepsArrProb),ArrProblist[a], label=GListLabel[a])
    plt.xlabel('steps')
    plt.ylabel('Arrival Probability')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.7)

    plt.show()