import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.linalg
from tqdm import tqdm


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

    return G,H,pos


def SDevHexRad(H,HexX,HexY,pos,stepsTot):

    SDev = np.zeros(stepsTot)
    xPosns = np.zeros(len(H[0]))
    yPosns = np.zeros(len(H[0]))
    RadialDist = np.zeros(len(H[0]))
    middleX = int(HexX/2)
    middleY = int(HexY)

    for k,v in pos.items():
        if np.around(v[0]) == middleX and np.around(v[1]) == middleY:
            initPosn = k

    for i in range(len(H[0])):
        xPosns[i] = pos[i][0]
        yPosns[i] = pos[i][1]/2
        RadialDist[i] = np.sqrt((abs(xPosns[i]-middleX)**2)+(abs(yPosns[i]-middleY)**2))

    for step in (range(stepsTot)):
        U = scipy.linalg.expm(-1j*H*step)
        psi0 = np.zeros(len(H[0]))
        psi0[initPosn] = 1 # initialise in the middle
        psiN = np.dot(U,psi0)

        probs = abs(psiN**2)

        AvgX = sum(RadialDist*probs)
        AvgXsq = sum((RadialDist**2)*probs)

        StandDev = np.sqrt(np.around((AvgXsq - (AvgX)**2),decimals=10)) # if not rounded the first value in the sqrt is negative (order e-16)
        SDev[step] = StandDev

    return SDev, probs



if __name__ == '__main__':

    HexX = 5
    HexY = 2
    HexYlabel = HexY
    stepsTot = 20
    trialsTot = 4
    SDtrials = 100

    SDevRadAll = []

    for t in tqdm(range(trialsTot)):

        G,H,pos = HexLattice(HexX,HexY)

        SDavg = np.zeros(stepsTot)
        
        for _ in tqdm(range(SDtrials)):
            SDev, probs = SDevHexRad(H,HexX,HexY,pos,stepsTot)
            SDavg += SDev
        
        SDavg = SDavg/SDtrials
        
        PlotProbs = probs

        SDevRadAll.append(SDavg)

        HexY += 1


    # plot

    fig = plt.figure()
    gs1 = gridspec.GridSpec(6, 5)
    gs1.update(hspace=0.3)


    ax2 = fig.add_subplot(gs1[:4,1:4])
    node_size = PlotProbs*(100000/(max(PlotProbs)*len(H[0])))
    PltNodes = nx.draw_networkx_nodes(G, pos, node_color=PlotProbs, node_size=node_size, with_label=False)
    PltEdges = nx.draw_networkx_edges(G, pos)
    col1 = fig.colorbar(PltNodes, label='Probability', shrink=0.9)


    gs2 = gridspec.GridSpec(6, 5)
    gs2.update(hspace=0.3)

    ax5 = fig.add_subplot(gs2[4:,:])
    for j in range(trialsTot):
        plt.plot(np.arange(stepsTot),SDevRadAll[j],label=('Hexagons in Y: '+str(HexYlabel+j)))
    plt.xlabel('steps')
    plt.ylabel('$\sigma_r$')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()
