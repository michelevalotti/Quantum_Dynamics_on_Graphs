import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.linalg
from tqdm import tqdm 

# M and N are number of HEXAGONS (not lattice points) in x and y

N = 1 # y
M = 6 # x - cylinder only works if this is even
gamma = 1.0
steps = 7
InitialPosn1 = 5 #13
InitialPosn2 = 20 #20

G = nx.hexagonal_lattice_graph(N,M)

nodes = G.number_of_nodes()
print('number of nodes: ', nodes)
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
Adj = Adj.astype(float)

# # join top and bottom edge (horizontal cylinder) -- when calculating the distance this is not considered

# for key,value in labels.items():
#     if (key[1] == 0 or ((key[1] == 1) and (key[0] != 0))):
#         Adj[value+(2*N), value] = 1
#         Adj[value, value+(2*N)] = 1



# quantum mechanics

Adj = np.kron(Adj,Adj)

Deg = np.zeros((nodes,nodes)) # degree matrix
Deg = np.kron(Deg,Deg)
for i in range(nodes):
    Deg[i,i] = np.sum(Adj[i,:])

for k in range(nodes-1,-1,-1):
	Adj = np.delete(Adj, (k*(nodes+1)), axis=0)
	Adj = np.delete(Adj, (k*(nodes+1)), axis=1)
	Deg = np.delete(Deg, (k*(nodes+1)), axis=0)
	Deg = np.delete(Deg, (k*(nodes+1)), axis=1)

AdjSize = len(Adj)

print('Adj done')

# distance - cartesian

PosList = []
for key, val in pos.items():
    PosList.append(pos[key])
PosDict = dict(zip(coordVal,PosList))

DistDict = dict( ((int(n/nodes), n%nodes), n) for n in range(nodes**2) if int(n/nodes)!=(n%nodes))
for key,val in DistDict.items():
    DistDict[key] = np.sqrt((abs(PosDict[key[0]][0]-PosDict[key[1]][0])**2)+(abs(PosDict[key[0]][1]-PosDict[key[1]][1])**2))

Distances = np.zeros(AdjSize)
b=0
for key, val in DistDict.items():
    Distances[b] = DistDict[key]
    b+=1

print('distances calculated')

# attraction or repulsion (change DistMx values)

maxDist = max(Distances)
DistSectors = [maxDist/3, (2*maxDist)/3, maxDist]

DistMx = np.zeros((AdjSize,AdjSize))
for n in range(AdjSize):
	for m in range(AdjSize):
		if ((Distances[m] <= DistSectors[0]) and (Distances[n] <= DistSectors[0])):
			DistMx[m,n] = 1.0
		elif (Distances[m] < DistSectors[0] and Distances[n] >= DistSectors[0]) or (Distances[m] >= DistSectors[0] and Distances[n] < DistSectors[0]):
			DistMx[m,n] = 0.1
		elif (((Distances[m] > DistSectors[0]) and (Distances[n] > DistSectors[0])) and ((Distances[m] <= DistSectors[1]) and (Distances[n] <= DistSectors[1]))):
			DistMx[m,n] = 0.3
		elif (Distances[m] < DistSectors[1] and Distances[n] >= DistSectors[1]) or (Distances[m] >= DistSectors[1] and Distances[n] < DistSectors[1]):
			DistMx[m,n] = 0.1
		elif (((Distances[m] > DistSectors[1]) and (Distances[n] > DistSectors[1])) and ((Distances[m] <= DistSectors[2]) and (Distances[n] <= DistSectors[2]))):
			DistMx[m,n] = 0.1
		elif (PosList[m][0] == PosList[n][1]) and (PosList[m][1] == PosList[n][0]): # particles exchange position
			DistMx[m,n] = 1.0

for i in range(AdjSize):
	for j in range(AdjSize):
		if Adj[i,j] == 1:
			Adj[i,j] = DistMx[i,j]


# evolve the system

H = np.zeros((AdjSize,AdjSize),dtype=complex)
H += gamma*(Deg-Adj)

meanDist = np.zeros(steps)
runningAvg = np.zeros(steps)

for step in tqdm(range(steps)):
	U = scipy.linalg.expm(-1j*H*step) # performance bottleneck

	psi0_1 = np.zeros(nodes)
	psi0_1[InitialPosn1] = 1
	psi0_2 = np.zeros(nodes)
	psi0_2[InitialPosn2] = 1
	psi0 = (np.kron(psi0_1,psi0_2) - np.kron(psi0_2,psi0_1))/(np.sqrt(2))
	# psi0 = np.kron(psi0_1, psi0_2)
	for k in range(nodes-1,-1,-1):
		psi0 = np.delete(psi0, (k*(nodes+1)))
	psiN = np.dot(U,psi0)

	weights = abs(psiN**2)
	weightedDist = weights*Distances

	meanDist[step] = sum(weightedDist)/(sum(weights))

	runningAvg[step] = (sum(meanDist[:step+1]))/(step+1)


# measure

probs = np.zeros(AdjSize)
prob2 = np.zeros(nodes)


for m in range(AdjSize):

    meas = np.zeros(AdjSize)
    meas[m] = 1
    probs[m] = abs(np.dot(psiN.T,meas))**2

prob1 = np.add.reduceat(probs, np.arange(0, len(probs), (nodes-1)))

# dictionary of (label1and2, prob)
ProbDict = dict( ((int(n/nodes), n%nodes), n) for n in range(nodes**2) if int(n/nodes)!=(n%nodes))
i=0
for key, val in ProbDict.items():
    ProbDict[key] = probs[i]
    i+=1

# find prob2 using dictionary
prob2 = np.zeros(nodes)
for i in range(nodes):
    for key, val in ProbDict.items():
        if key[1] == i:
            prob2[i] += val

# check normalizations
print('particle 1 tot prob: ', sum(prob1))
print('particle 2 tot prob: ', sum(prob2))


# edge colors

EdgeList = []
for i in G.edges():
    EdgeList.append(i)

ProbsDict1 = dict(zip(coords,prob1)) # dictionary of probabilities associated with coordinates
edge_colors1 = np.zeros((edges,2))

for i in range(len(EdgeList)):
    for key,val in ProbsDict1.items():
        if EdgeList[i][0] == key:
            edge_colors1[i][0] = val
        if EdgeList[i][1] == key:
            edge_colors1[i][1] = val

edge_colors1 = np.average(edge_colors1,axis=1)


ProbsDict2 = dict(zip(coords,prob2)) # dictionary of probabilities associated with coordinates
edge_colors2 = np.zeros((edges,2))

for i in range(len(EdgeList)):
    for key,val in ProbsDict2.items():
        if EdgeList[i][0] == key:
            edge_colors2[i][0] = val
        if EdgeList[i][1] == key:
            edge_colors2[i][1] = val

edge_colors2 = np.average(edge_colors2,axis=1)


# draw

fig = plt.figure(figsize=(12,6), dpi=200)

gs1 = gridspec.GridSpec(3, 3)
gs1.update(left=0.1, right=1.0, wspace=0.0, hspace=0.0)

ax1 = fig.add_subplot(gs1[0,:])
node_size1 = prob1*(10000/(max(prob1)*nodes))
# nx.draw(G, pos=pos, labels=labels, edge_color=edge_colors)
PltNodes = nx.draw_networkx_nodes(G, pos, node_color=prob1, node_size=node_size1, with_label=False)
PltEdges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors1)
cbar1 = fig.colorbar(PltNodes, label='Particle 1 Prob', shrink=0.9, format='%.2f', ticks=np.linspace(min(prob1), max(prob1), 6))
cbar1.set_label('Particle 1 Prob',labelpad=20)
ax1.set_yticklabels([])
plt.yticks([])
ax1.set_xticklabels([])
plt.xticks([])

ax2 = fig.add_subplot(gs1[1,:])
node_size2 = prob2*(10000/(max(prob2)*nodes))
# nx.draw(G, pos=pos, labels=labels, edge_color=edge_colors)
PltNodes = nx.draw_networkx_nodes(G, pos, node_color=prob2, node_size=node_size2, with_label=False)
PltEdges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors2)
cbar2 = fig.colorbar(PltNodes, label='Particle 2 Prob', shrink=0.9, format='%.2f', ticks=np.linspace(min(prob2), max(prob2), 6))
cbar2.set_label('Particle 2 Prob',labelpad=20)
ax2.set_yticklabels([])
plt.yticks([])
ax2.set_xticklabels([])
plt.xticks([])


gs2=gridspec.GridSpec(3,3)
gs2.update(left=0.1, right=0.82, wspace=0.0)

ax3 = fig.add_subplot(gs2[2,:])
xAx = np.arange(steps)
plt.plot(xAx, runningAvg)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position('right')
ax3.yaxis.labelpad = 12
plt.xlabel('steps')
plt.ylabel('Avg Distance')


plt.show()