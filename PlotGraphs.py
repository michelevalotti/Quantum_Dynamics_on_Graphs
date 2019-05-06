import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from tqdm import tqdm



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
        for _ in range(NextConns):
            NextNode1 = np.random.randint(C*ClusterNodes, (C+1)*ClusterNodes)
            NextNode2 = np.random.randint((C+1)*ClusterNodes, (C+2)*ClusterNodes)
            while(int(NextNode1/ClusterNodes) == int(NextNode2/ClusterNodes)):
                NextNode2 =  np.random.randint((C+1)*ClusterNodes, (C+2)*ClusterNodes)

            G.add_edge(NextNode1,NextNode2)

    return G

def ConnectHexNext(G, HexY, NextConns, Clusters, orientation, pos):

    ClusterNodes = G.number_of_nodes()/Clusters

    if orientation == 'horizontal':
        for C in range(Clusters-1):
            for _ in range(NextConns):
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
            for _ in range(NextConns):
                NextNode1 = random.choice(TopNodes)
                NextNode2 = random.choice(BottomNodes)
                G.add_edge(NextNode1,NextNode2)

    return G

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



def ClusterGraph(Clusters,ClusterNodes,ClusterConnections,ClusterDist):
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



def HexLattice(HexX,HexY,gamma=1.0):

    G = nx.hexagonal_lattice_graph(HexY,HexX)

    nodes = G.number_of_nodes()

    pos = dict( (n, n) for n in G.nodes() ) # this gives positions on a square lattice


    coords = []
    for i,j in G.nodes():
        coords.append((i,j))
    coordVal = []
    for k in range(nodes):
        coordVal.append(k)


    # shift positions to make graph look like a hexagonal lattice

    for coord in range(len(coords)):
        if ((coords[coord][0]%2 == 0) and (coords[coord][1]%2 != 0)):
            coords[coord] = ((float(coords[coord][0]) - 0.15), coords[coord][1])
        elif ((coords[coord][0]%2 != 0) and (coords[coord][1]%2 != 0)):
            coords[coord] = ((float(coords[coord][0]) + 0.15), coords[coord][1])
        elif ((coords[coord][0]%2 == 0) and (coords[coord][1]%2 == 0)):
            coords[coord] = ((float(coords[coord][0]) + 0.15), coords[coord][1])
        elif ((coords[coord][0]%2 != 0) and (coords[coord][1]%2 == 0)):
            coords[coord] = ((float(coords[coord][0]) - 0.15), coords[coord][1])



    # adjacency matrix

    Adj = nx.adjacency_matrix(G) # positions are labelled from lower left corner up, every column starts at the bottom
    Adj = Adj.todense()

    pos = dict(zip(coordVal,coords))

    LabelDict = dict(zip(G.nodes(),coordVal)) # rename nodes from 0 to nodes

    G = nx.relabel_nodes(G,LabelDict)

    TotNodes = G.number_of_nodes()

    Deg = np.zeros((TotNodes,TotNodes))
    for i in range(TotNodes):
        Deg[i,i] = np.sum(Adj[i,:])

    H = np.zeros((TotNodes,TotNodes),dtype=complex)
    H += gamma*(Deg-Adj)

    return G,pos


# # hexagonal clusters

# fig = plt.figure(figsize=(14,10), dpi=200)

# G_noconn, pos = HexCluster(10,2,1,'vertical')

# ax1 = fig.add_subplot(141)
# nx.draw(G_noconn, pos, node_size=10)

# ax2 = fig.add_subplot(142)
# G_next = ConnectHexNext(G_noconn, 1, 6, 10, 'vertical', pos=pos)
# nx.draw(G_next, pos, node_size=10)

# ax3 = fig.add_subplot(143)
# G_allnext = ConnectAllNext(G_noconn, 6, 10)
# nx.draw(G_allnext, pos, node_size=10)

# ax4 = fig.add_subplot(144)
# G_rand = ConnectRand(G_noconn, 60, 10)
# nx.draw(G_rand, pos, node_size=10)


# # random clusters

# fig = plt.figure(figsize=(14,3), dpi=200)

# G_rand_noconn, pos_cluster = ClusterGraph(5, 20, 30, 20)
# GC_next = ConnectAllNext(G_rand_noconn,3,5)
# GC_rand = ConnectRand(G_rand_noconn,12,5)

# nx.draw(GC_rand, pos=pos_cluster, node_size=10)


# hexagonal lattice

fig = plt.figure(figsize=(10,10), dpi=200)


G_hex, pos_hex = HexLattice(6,6)

nx.draw(G_hex, pos=pos_hex, node_size=10)

plt.show()