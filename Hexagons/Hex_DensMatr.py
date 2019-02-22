import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.linalg
from tqdm import tqdm

# M and N are number of HEXAGONS (not lattice points) in x and y

# for density matrix reproduce commutator and don't forget to multiply by dt

N = 8 # y
M = 8 # x - cylinder only works if this is even
gamma = 1.0
steps = 20
InitialPosn = 80
decRate = 1.0
dt = 0.06 # timestep for numerical integration

G = nx.hexagonal_lattice_graph(N,M)

nodes = G.number_of_nodes()
print(nodes)
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


# # join top and bottom edge (horizontal cylinder)

# for key,value in labels.items():
#     if (key[1] == 0 or ((key[1] == 1) and (key[0] != 0))):
#         Adj[value+(2*N), value] = 1
#         Adj[value, value+(2*N)] = 1


# degree matrix

Deg = np.zeros((nodes,nodes))
for i in range(nodes):
    Deg[i,i] = np.sum(Adj[i,:])


# distance from InitialPosn or last node (CoordsDics[nodes-1])

CoordsDict = dict(zip(coordVal,coords))
Distances = np.zeros(nodes)
for key,val in CoordsDict.items():
    Distances[key] = np.sqrt((abs(CoordsDict[InitialPosn][0]-CoordsDict[key][0])**2)+(abs(CoordsDict[InitialPosn][1]-CoordsDict[key][1])**2))


# quantum mechanics

H = np.zeros((nodes,nodes),dtype=complex)

H += gamma*(Deg-Adj)

# # add potential
# Pot = np.zeros((nodes,nodes))
# for key, val in labels.items():
#     Pot[-val-1,-val-1] = np.round(key[0])/10 # Pot[val,val] if values increase left to right, Pot[-val-1,-val-1] if they increase right to left

# H += Pot

# H[82,82] -= 1000

meanDist = np.zeros(steps)
runningAvg = np.zeros(steps)

psi0 = np.zeros(nodes,dtype=complex)
psi0[InitialPosn] = 1
psiN = np.outer(psi0,psi0)
# k = 0

for step in tqdm(range(steps)):

	# decoherence

    decOp = np.zeros((nodes,nodes), dtype=complex)

    for j in range(nodes):
        proj = np.zeros(nodes, dtype=complex)
        proj[j] = 1
        projMatr = np.outer(proj,proj)
        decoherence = np.dot(projMatr, psiN)
        decoherence = np.dot(decoherence, projMatr.T)
        decOp += decoherence


    # psiN += (np.dot(U,psiN)-np.dot(psiN,U)) - (decRate)*psiN + (decRate)*decOp

    # psiN += (1 - decRate*dt)*(psiN-(1j*gamma*dt*(np.dot(H,psiN)- np.dot(psiN.T,H)))) + decRate*dt*decOp
    psiN += -(1j*gamma*dt*(np.dot(H,psiN)- np.dot(psiN.T,H))) - (decRate*dt*psiN) + (dt*decRate*decOp)

    # U = scipy.linalg.expm(-1j*H*k)

    # psiN_0 = np.dot(U,psiN)
    # psiN_0 = np.dot(psiN_0,U.T)

    # psiN_0 = 

    # psiN += psiN_0 + psiN_1


    # k += dt

    # # calculating avg distance is not very useful if the system doesn't evolve after it hits the boarder of the graph
    # # higher decRate means lower avg dist for same number of steps (before reacing edge)
    # # once psiN reaches the edge probability starts to leave the system (not numerical integration error)
    # weights = np.zeros(nodes)

    # for i in range(nodes):

    #     meas = np.zeros(nodes)
    #     meas[i] = 1
    #     RhoMeas = np.outer(meas,meas.T)
    #     weights[i] = np.absolute(np.trace(np.dot(psiN,RhoMeas)))

    # weightedDist = weights*Distances

    # meanDist[step] = sum(weightedDist)

    # runningAvg[step] = (sum(meanDist[:step+1]))/(step+1)


# measure

probs = np.zeros(nodes)

for i in range(nodes):

    meas = np.zeros(nodes)
    meas[i] = 1
    RhoMeas = np.outer(meas,meas.T)
    probs[i] = np.absolute(np.trace(np.dot(psiN,RhoMeas)))

# check normalizations
print(sum(probs))

ProbsDict = dict(zip(coords,probs)) # dictionary of probabilities associated with coordinates

# probability plots on the size of graph
probsX = np.zeros(M+1)
probsY = np.zeros(N*2 + 2)

for i in range(M+1):
    for key, val in ProbsDict.items():
        if np.round(key[0]) == i:
            probsX[i] += val

for i in range(N*2 + 2):
    for key, val in ProbsDict.items():
        if np.round(key[1]) == i:
            probsY[i] += val


# edge colors

EdgeList = []
for i in G.edges():
    EdgeList.append(i)

edge_colors = np.zeros((edges,2))

for i in range(len(EdgeList)):
    for key,val in ProbsDict.items():
        if EdgeList[i][0] == key:
            edge_colors[i][0] = val
        if EdgeList[i][1] == key:
            edge_colors[i][1] = val

edge_colors = np.average(edge_colors,axis=1)


# draw

xAxProbsX = np.arange(M+1)
xAxProbsY = np.arange(N*2+2)

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], wspace=0.0, hspace=0.0)

ax1 = fig.add_subplot(gs[1])

node_size = probs*(100000/(max(probs)*nodes))
# nx.draw(G, pos=pos, labels=labels, edge_color=edge_colors)
PltNodes = nx.draw_networkx_nodes(G, pos, node_color=probs, node_size=node_size, with_label=False)
PltEdges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
plt.colorbar(PltNodes, label='Probability', shrink=0.9)

ax2 = fig.add_subplot(gs[0])
plt.plot(probsY, xAxProbsY)
plt.xlabel('$P_{x,tot}$')
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top') 

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], wspace=0.0, hspace=0.0, right=0.776, left=0.156)
ax3 = fig.add_subplot(gs[3])
plt.plot(xAxProbsX, probsX)
plt.ylabel('$P_{y,tot}$', rotation=0)
ax3.yaxis.labelpad = 12
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position('right')

plt.show()