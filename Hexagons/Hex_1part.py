import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.linalg
from tqdm import tqdm

# M and N are number of HEXAGONS (not lattice points) in x and y
# if M >> N calculates HorDist, if M << N calculates VertDist
# choose values of M and N so that graph is long and thin

N = 30 # y
M = 3 # x - horizontal cylinder (M > N) only works if this is even
gamma = 1.0
steps = 100

G = nx.hexagonal_lattice_graph(N,M)

nodes = G.number_of_nodes()
edges = G.number_of_edges()

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

for key,value in pos.items():
    if ((key[0]%2 == 0) and (key[1]%2 != 0)):
        pos[key] = ((float(value[0]) - 0.15), value[1])
    if ((key[0]%2 != 0) and (key[1]%2 != 0)):
        pos[key] = ((float(value[0]) + 0.15), value[1])
    if ((key[0]%2 == 0) and (key[1]%2 == 0)):
        pos[key] = ((float(value[0]) + 0.15), value[1])
    if ((key[0]%2 != 0) and (key[1]%2 == 0)):
        pos[key] = ((float(value[0]) - 0.15), value[1])



# adjacency matrix

Adj = nx.adjacency_matrix(G) # positions are labelled from lower left corner up, every column starts at the bottom
Adj = Adj.todense()

if M > N:
    # join top and bottom edge (horizontal cylinder)
    for key,value in labels.items():
        if (key[1] == 0 or ((key[1] == 1) and (key[0] != 0))):
            Adj[value+(2*N), value] = 1
            Adj[value, value+(2*N)] = 1

if M < N:
    # join left and right sides of the cylinder (vertical tube)
    for key, value in labels.items():
        if (key[0] == 0 and (key[1] != 0 and key[1] != (2*N+1))):
            for key1, value1 in labels.items():
                if (key1[1] == key[1] and (key1[0] == (key[0] + N))):
                    Adj[value, value1] = 1
                    Adj[value1, value] = 1


# degree matrix

Deg = np.zeros((nodes,nodes))
for i in range(nodes):
    Deg[i,i] = np.sum(Adj[i,:])


# distance from end of the lattice

CoordsDict = dict(zip(coordVal,coords))

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


meanDist = np.zeros(steps)
runningAvg = np.zeros(steps)
EndProb = np.zeros(steps)

for step in tqdm(range(steps)):
    U = scipy.linalg.expm(-1j*H*step)
    psi0 = np.zeros(nodes)
    if M < N:
        for key, val in labels.items():
            if (key[1] == 0 or key[1] == 1): # particle starts at the bottom (vertical cylinder)
                psi0[val] = 1 # superposition of all nodes on the left
    if M > N:
        for key, val in labels.items():
            if (key[0] == 0 or key[0] == 1): # particle starts on the left (horizontal cylinder)
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

    runningAvg[step] = (sum(meanDist[:step+1]))/(step+1)

    # calculate probability at the end of the tube
    if M < N:
        for key, val in labels.items():
            if (key[1] == 2*N+1 or key[1] == 2*N):
                EndProb[step] += weights[val]
    if M > N:
        for key, val in labels.items():
            if (key[0] == M):
                EndProb[step] += weights[val]


# measure

probs = np.zeros(nodes)

for i in range(nodes):

	meas = np.zeros(nodes)
	meas[i] = 1
	probs[i] = abs(np.dot(psiN.T,meas))**2

# check normalizations
print(sum(probs))


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


# draw

xAx = np.arange(steps)

if M < N: # vertical graph
    fig = plt.figure()
    gs1 = gridspec.GridSpec(10, 3)
    gs1.update(hspace=0.0)

    ax1 = fig.add_subplot(gs1[:,2])

    node_size = probs*(100000/(max(probs)*nodes))
    # nx.draw(G, pos=pos, labels=labels, edge_color=edge_colors)
    PltNodes = nx.draw_networkx_nodes(G, pos, node_color=probs, node_size=node_size, with_label=False)
    PltEdges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
    plt.colorbar(PltNodes, label='Probability', shrink=0.9)

    gs2=gridspec.GridSpec(10,3)
    gs2.update(left=0.1, right=0.85, hspace=0.0)

    ax2 = fig.add_subplot(gs2[1:5,:2])
    plt.xlabel('steps')
    plt.ylabel('Prob Last state')
    plt.plot(xAx, EndProb)

    ax3 = fig.add_subplot(gs2[5:9,:2])
    plt.xlabel('steps')
    plt.ylabel('avg distance from top')
    plt.plot(xAx, runningAvg)

if M > N: # horizontal graph
    fig = plt.figure()
    gs1 = gridspec.GridSpec(10, 3)
    gs1.update(hspace=20.0)

    ax1 = fig.add_subplot(gs1[0:4,:])

    node_size = probs*(100000/(max(probs)*nodes))
    # nx.draw(G, pos=pos, labels=labels, edge_color=edge_colors)
    PltNodes = nx.draw_networkx_nodes(G, pos, node_color=probs, node_size=node_size, with_label=False)
    PltEdges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
    plt.colorbar(PltNodes, label='Probability', shrink=0.9)

    gs2=gridspec.GridSpec(10,3)
    gs2.update(left=0.125, right=0.745, hspace=0.0)

    ax2 = fig.add_subplot(gs2[4:7,:])
    plt.xlabel('steps')
    plt.ylabel('Prob Last state')
    plt.plot(xAx, EndProb)

    ax3 = fig.add_subplot(gs2[7:,:])
    plt.xlabel('steps')
    plt.ylabel('avg distance from end')
    plt.plot(xAx, runningAvg)

plt.show()