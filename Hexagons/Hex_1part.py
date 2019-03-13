import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.linalg
from tqdm import tqdm


def HexTube(HexX,HexY,ChiralShift):

    G = nx.hexagonal_lattice_graph(HexY,HexX)

    nodes = G.number_of_nodes()
    print(nodes)

    pos = dict( (n, n) for n in G.nodes() ) # this gives positions on a square lattice


    # label dictionary for debugging and to make cylinder (see adjacency matrix)

    coords = []
    for i,j in G.nodes():
        coords.append((i,j))
    coordVal = []
    for k in range(nodes):
        coordVal.append(k)

    labels = dict(zip(coords, coordVal))



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

    if HexX > HexY:
        # join top and bottom edge (horizontal cylinder)
        if ChiralShift != 0:
            for key,value in labels.items():
                if (key[1] == 0 or key[1] == 1):
                    for k,v in labels.items():
                        if (k[1] == key[1]+(2*HexY)) and (k[0] == key[0]+(2*ChiralShift)):
                            Adj[v, value] = 1
                            Adj[value, v] = 1
        else:
            for key,value in labels.items():
                if (key[1] == 0 or ((key[1] == 1) and (key[0] != 0))):
                    Adj[value+(2*HexY), value] = 1
                    Adj[value, value+(2*HexY)] = 1

    if HexX < HexY:
        # join left and right sides of the cylinder (vertical tube)
        for key, value in labels.items():
            if (key[0] == 0 and (key[1] != 0 and key[1] != (2*HexY+1))):
                for key1, value1 in labels.items():
                    if (key1[1] == key[1] and (key1[0] == (key[0] + HexY))):
                        Adj[value, value1] = 1
                        Adj[value1, value] = 1

    pos = dict(zip(coordVal,coords))

    LabelDict = dict(zip(G.nodes(),coordVal)) # rename nodes from 0 to nodes

    G = nx.relabel_nodes(G,LabelDict)

    return G,Adj,pos


def ArrivalProbability(G,M,N,Adj,pos,steps,eta=1.0,gamma=1.0):

    edges = G.number_of_edges()
    nodes = G.number_of_nodes()
    
    coords = []
    coordVal = []
    for k,v in pos.items():
        coords.append(v)
        coordVal.append(k)

    

    # degree matrix

    Deg = np.zeros((nodes,nodes))
    for i in range(nodes):
        Deg[i,i] = np.sum(Adj[i,:])


    # distance from end of the lattice

    CoordsDict = dict(zip(coordVal,coords))
    labels = dict(zip(coords, coordVal))


    HorDist = np.zeros(nodes) # horizontal distance
    VertDist = np.zeros(nodes) # vertical distance
    for key,val in CoordsDict.items():
        HorDist[key] = abs(M-CoordsDict[key][0]) # distance from end of tube (on right)
        VertDist[key] = abs((2*N + 1)-CoordsDict[key][1]) # distance from top of the tube(y corods of 2N+1)


    # quantum mechanics

    H = np.zeros((nodes,nodes),dtype=complex)

    H += gamma*(Deg-Adj)

    # # add potential -- simple model of defects (high and low potentials have the same effect)
    # Pot = np.zeros((nodes,nodes))
    # for key, val in labels.items():
    #     Pot[-val-1,-val-1] = np.round(key[0])/10 # Pot[val,val] if values increase left to right, Pot[-val-1,-val-1] if they decrease left to right
    # H += Pot
    # H[110,110] -= 10
    # ########


    # sites on the top or right (according to geometry of graph) have high losses
    # when the particle arrives at the opposite side of the graph it exits the system

    if M < N:
        for key, val in labels.items():
            if (int(np.round(key[1])) == 2*N+1 or int(np.round(key[1])) == 2*N):
                H[val,val] -= 1j*(eta/2)

    if M > N:
        for key, val in labels.items():
            if (int(np.round(key[0])) == M):
                H[val,val] -= 1j*(eta/2)

    ###########


    meanDist = np.zeros(steps)
    runningAvg = np.zeros(steps)
    EndProb = np.zeros(steps)

    for step in tqdm(range(steps)):
        U = scipy.linalg.expm(-1j*H*step)
        psi0 = np.zeros(nodes)
        if M < N:
            for key, val in labels.items():
                if (int(np.round(key[1])) == 0 or int(np.round(key[1])) == 1): # particle starts at the bottom (vertical cylinder)
                    psi0[val] = 1 # superposition of all nodes on the bottom
        if M > N:
            for key, val in labels.items():
                if (int(np.round(key[0])) == 0 or int(np.round(key[0])) == 1): # particle starts on the left (horizontal cylinder)
                    psi0[val] = 1 # superposition of all nodes on the left
        psi0 = psi0/(np.sqrt(sum(psi0)))
        psiN = np.dot(U,psi0)


        weights = abs(psiN**2)

        # HorDist of VertDist depending on shape of the graph  
        if M < N:
            weightedDist = weights*VertDist
        if M > N:
            weightedDist = weights*HorDist

        meanDist[step] = sum(weightedDist)

        if M < N:
            runningAvg[step] = 2*N - (sum(meanDist[:step+1]))/(step+1)
        if M > N:
            runningAvg[step] = M - (sum(meanDist[:step+1]))/(step+1)

        if eta == 0.0:
            # calculate probability at the end of the tube
            if M < N:
                for key, val in labels.items():
                    if (int(np.round(key[1])) == 2*N+1 or int(np.round(key[1])) == 2*N):
                        EndProb[step] += weights[val]
            if M > N:
                for key, val in labels.items():
                    if (int(np.round(key[0])) == M):
                        EndProb[step] += weights[val]
        
        else:
            EndProb[step] = 1 - sum(weights)


    # measure

    probs = np.zeros(nodes)

    for i in range(nodes):

        meas = np.zeros(nodes)
        meas[i] = 1
        probs[i] = abs(np.dot(psiN.T,meas))**2


    # edge colors

    EdgeList = []
    for i in G.edges():
        EdgeList.append(i)

    ProbsDict = dict(zip(coords,probs)) # dictionary of probabilities associated with coordinates
    edge_colors = np.zeros((edges,2))

    for i in range(len(EdgeList)):
        for key,val in ProbsDict.items():
            if EdgeList[i][0] == key:
                edge_colors[i][0] = val
            if EdgeList[i][1] == key:
                edge_colors[i][1] = val

    edge_colors = np.average(edge_colors,axis=1)

    return probs, EndProb, runningAvg, edge_colors, H


def StandardDeviationHex(G, M, N, Adj, pos, steps, gamma=1.0):

    TotNodes = G.number_of_nodes()

    Deg = np.zeros((TotNodes,TotNodes))
    for i in range(TotNodes):
        Deg[i,i] = np.sum(Adj[i,:])

    H = np.zeros((TotNodes,TotNodes),dtype=complex)
    H += gamma*(Deg-Adj)

    StandDevArr = np.zeros(steps)

    for step in tqdm(range(steps)):
        U = scipy.linalg.expm(-1j*H*step)
        psi0 = np.zeros(TotNodes, dtype=complex)
        xPosns = np.zeros(TotNodes)
        yPosns = np.zeros(TotNodes)

        MiddleX = (int((abs((pos[0])[0] - (pos[M])[0])/2)))
        MiddleY = (int((abs((pos[0])[1] - (pos[N*2])[1])/2)))

        if M > N:
            for key, val in pos.items():
                xPosns[key] = val[0]
                if (val[0] >= (MiddleX) and val[0] <= (MiddleX+1)): # particle starts in the middle
                    psi0[key] = 1 # superposition of all nodes in the middle
            psi0 = psi0/(np.sqrt(sum(psi0)))
            psiN = np.dot(U,psi0)

            probs = abs(psiN**2)

            AvgX = sum(xPosns*probs)
            AvgXsq = sum((xPosns**2)*probs)

            StandDev = np.sqrt((AvgXsq - (AvgX)**2))
            StandDevArr[step] = StandDev
        
        if M < N:
            for key, val in pos.items():
                yPosns[key] = val[1]
                if (val[1] >= (MiddleY) and val[1] <= (MiddleY+1)): # particle starts in the middle
                    psi0[key] = 1 # superposition of all nodes in the middle
            psi0 = psi0/(np.sqrt(sum(psi0)))
            psiN = np.dot(U,psi0)

            probs = abs(psiN**2)

            AvgX = sum(yPosns*probs)
            AvgXsq = sum((yPosns**2)*probs)

            StandDev = np.sqrt((AvgXsq - (AvgX)**2))
            StandDevArr[step] = StandDev

    return StandDevArr, probs


if __name__ == "__main__":

    # M and N are number of HEXAGONS (not lattice points) in x and y
    # if M >> N calculates HorDist, if M << N calculates VertDist
    # choose values of M and N so that graph is long and thin

    # add loss site at end of the tube and record probability that leaves the system

    N = 20 # y
    M = 2 # x - horizontal cylinder (M > N) only works if this is even
    gamma = 1.0
    steps = 10
    stepsSDev = int(steps) # reaches maximum deviation quicker than ArrivalProb, but is also interesting to see long term behaviour
    eta = 1.0 # loss rate
    ChiralShift = 1 # wraps tube around with a shift of 2*ChiralShift units in the x (skips one hexagon)

    MyGraph = HexTube(M,N,ChiralShift)

    G = MyGraph[0]
    Adj = MyGraph[1]
    pos = MyGraph[2]

    myValues = ArrivalProbability(G,M,N,Adj,pos,steps,eta)

    nodes = G.number_of_nodes()
    probs = myValues[0]
    EndProb = myValues[1]
    runningAvg = myValues[2]
    edge_colors = myValues[3]
    H = myValues[4]

    mySpreading = StandardDeviationHex(G,M,N,Adj,pos,stepsSDev)
    SDev = mySpreading[0]

    EigenVals, EigenVecs = np.linalg.eig(H)
    for i in range(len(EigenVals)):
        if abs(EigenVals[i]**2) == (EigenVals[i]**2):
            print(EigenVals[i])


    # draw

    xAx = np.arange(steps)

    if M < N: # vertical graph
        fig = plt.figure(figsize=(7,8), dpi=200)
        gs1 = gridspec.GridSpec(11, 3)
        gs1.update(hspace=0.0, right=0.85)

        ax1 = fig.add_subplot(gs1[:,2])

        node_size = probs*(100000/(max(probs)*nodes))
        PltNodes = nx.draw_networkx_nodes(G, pos, node_color=probs, node_size=node_size, with_label=False)
        PltEdges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
        col1 = fig.colorbar(PltNodes, label='Probability', shrink=0.9)
        col1.ax.tick_params(labelsize=10)
        ax1.tick_params(labelsize=10)

        gs2=gridspec.GridSpec(11,3)
        gs2.update(left=0.1, right=0.85, hspace=0.0)

        ax2 = fig.add_subplot(gs2[1:4,:2])
        plt.xlabel('steps', fontsize=12)
        plt.ylabel('Avg Position', fontsize=12)
        plt.plot(xAx, runningAvg)
        ax2.tick_params(labelsize=12)

        ax3 = fig.add_subplot(gs2[4:7,:2])
        plt.xlabel('steps', fontsize=12)
        plt.ylabel('$P_{arrival}$', fontsize=12)
        plt.plot(xAx, EndProb)
        ax3.tick_params(labelsize=12)

        ax4 = fig.add_subplot(gs2[7:10,:2])
        plt.xlabel('steps', fontsize=12)
        plt.ylabel('$\sigma_y$', fontsize=12)
        plt.plot(np.arange(stepsSDev), SDev)
        ax4.tick_params(labelsize=12)


    if M > N: # horizontal graph
        fig = plt.figure(figsize=(7,8), dpi=200)
        gs1 = gridspec.GridSpec(10, 3)
        gs1.update(hspace=20.0)

        ax1 = fig.add_subplot(gs1[0:4,:])

        node_size = probs*(100000/(max(probs)*nodes))
        PltNodes = nx.draw_networkx_nodes(G, pos, node_color=probs, node_size=node_size, with_label=False)
        PltEdges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
        col1 = fig.colorbar(PltNodes, label='Probability', shrink=0.9)
        col1.ax.tick_params(labelsize=10)
        ax1.tick_params(labelsize=10)


        gs2=gridspec.GridSpec(10,3)
        gs2.update(left=0.125, right=0.745, hspace=0.0)

        ax2 = fig.add_subplot(gs2[4:6,:])
        plt.xlabel('steps', fontsize=12)
        plt.ylabel('Avg Position', fontsize=12)
        plt.plot(xAx, runningAvg)
        ax2.tick_params(labelsize=12)

        ax3 = fig.add_subplot(gs2[6:8,:])
        plt.xlabel('steps', fontsize=12)
        plt.ylabel('$P_{arrival}$', fontsize=12)
        plt.plot(xAx, EndProb)
        ax3.tick_params(labelsize=12)

        ax4 = fig.add_subplot(gs2[8:10,:])
        plt.xlabel('steps', fontsize=12)
        plt.ylabel('$\sigma_x$', fontsize=12)
        plt.plot(np.arange(stepsSDev), SDev)
        ax4.tick_params(labelsize=12)

    plt.show()