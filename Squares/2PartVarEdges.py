import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.linalg
from tqdm import tqdm # progress bar

N = 6 # x size of side of lattice (keep M*N<=36)
M = 6 # y size of lattice
gamma = 1.0 # hopping rate
steps = 0
init1 = 14 # initial position of part 1 (bottom left is 0, grows to the right and up, starts at left of every new row)
init2 = 21 # initial posn of part2
initialPosn = (init1*(N*M - 1)) + (init2-1) # element of initial vector with value one
stepsTot = 40 # steps quantum particles takes on lattice
eta = 0.5 # loss rate
LossSite = 13



# REFLECTIVE EDGES


# calculate distance between particles in each state
# and build dictionary of state to positions of particles

dist = np.zeros(((M*N)**2) - (N*M)) # array to store distances between particles in each two-particle state
positions = np.zeros((((M*N)**2) - (N*M), 2)) # vocabulary of positons of particles for each state

for p in range(len(dist)):
	posn1 = int(p/(N*M - 1))
	posn2 = p%(N*M - 1)
	if posn2 >= posn1:
		posn2 = posn2 + 1

	positions[p] = np.array([posn1,posn2])

	x1 = posn1%N
	y1 = int(posn1/N)
	x2 = posn2%N
	y2 = int(posn2/N)
	dist[p] = np.sqrt((x1-x2)**2 + (y1-y2)**2)

print('initial positions: ',positions[initialPosn])
print('initial distance: ',dist[initialPosn])
print('maximum distance: ',np.max(dist))


# build adj matrix for one particle, then kron product it with itself and remove columns/rows with 2 particles in the same state

Adj = np.zeros((N*M, N*M))#, dtype=complex) # cannot hop diagonally, enumerate 0-steps**2 starting top left and ending bottom right (going left to right, starting at left)
degree = np.eye((N*M)**2, dtype=complex)

for i in range(M*N): # i is y, j is x

	for j in range(M*N):

		if abs(i-j) == 1 or abs(i-j) == N:
			Adj[i,j] = 1
		if i%N==0:
			Adj[i,i-1] = 0
		if j%N==0:
			Adj[j-1,j] = 0

Adj = np.kron(Adj,Adj)

for i in range((N*M)**2):
	degree[i,i] = np.sum(Adj[i,:])


for k in range(N*M-1,-1,-1):
	Adj = np.delete(Adj, (k*(N*M+1)), axis=0)
	Adj = np.delete(Adj, (k*(N*M+1)), axis=1)
	degree = np.delete(degree, (k*(N*M+1)), axis=0)
	degree = np.delete(degree, (k*(N*M+1)), axis=1)



# PARTICLES REPEL EACH OTHER

# divide distance in 3 sectors, the probability of staying in a lower sector is lower than staying
# in higher one,  highest sector has highest probability, probability of jumping between sectors is high.
# Lowest probability when two states swap.

maxDist = np.max(dist)
DistSectors = [maxDist/3, (2*maxDist)/3, maxDist]

ProbMatr = np.zeros((((M*N)**2) - (N*M),((M*N)**2) - (N*M)))
for n in range(((M*N)**2) - (N*M)):
	for m in range(((M*N)**2) - (N*M)):
		if ((dist[m] <= DistSectors[0]) and (dist[n] <= DistSectors[0])):
			ProbMatr[m,n] = 0.1
		elif (dist[m] < DistSectors[0] and dist[n] >= DistSectors[0]) or (dist[m] >= DistSectors[0] and dist[n] < DistSectors[0]):
			ProbMatr[m,n] = 0.9
		elif (((dist[m] > DistSectors[0]) and (dist[n] > DistSectors[0])) and ((dist[m] <= DistSectors[1]) and (dist[n] <= DistSectors[1]))):
			ProbMatr[m,n] = 0.4
		elif (dist[m] < DistSectors[1] and dist[n] >= DistSectors[1]) or (dist[m] >= DistSectors[1] and dist[n] < DistSectors[1]):
			ProbMatr[m,n] = 0.9
		elif (((dist[m] > DistSectors[1]) and (dist[n] > DistSectors[1])) and ((dist[m] <= DistSectors[2]) and (dist[n] <= DistSectors[2]))):
			ProbMatr[m,n] = 1.0
		elif (positions[m][0] == positions[n][1]) and (positions[m][1] == positions[n][0]): # particles exchange position
			ProbMatr[m,n] = 0.0


for i in range(((M*N)**2) - (N*M)):
	for j in range(((M*N)**2) - (N*M)):
		if Adj[i,j] == 1:
			Adj[i,j] = ProbMatr[i,j]


######


H = gamma*(degree-Adj)

if np.allclose(H, H.T):
	print('Hamiltonian is symmetric')


# # SIMPLE MODEL OF LOSS

# # one site where there is prob particle 1 leaves the system, if it does, part2 also leaves
# # e.g. diatomic molecule bond breaks and atoms fly out

# for i in range(len(positions)):
# 	if positions[i][0] == LossSite:
# 		H[i,i] -= 1j*(eta/2)

# #######



meanDist = np.zeros(stepsTot)
runningAvg = np.zeros(stepsTot)


for r in tqdm(range(stepsTot)):

	U = scipy.linalg.expm(-1j*H*steps)

	psi0 = np.zeros(((M*N)**2) - (N*M))
	psi0[initialPosn] = 1
	psiN = np.dot(U,psi0)

	probs = np.zeros(((M*N)**2) - (N*M))
	prob2 = np.zeros(N*M)


	# # EVOLUTION AT EVERY STEP

	# for m in range(((M*N)**2) - (N*M)):

	# 	meas = np.zeros(((M*N)**2) - (N*M))
	# 	meas[m] = 1
	# 	probs[m] = abs(np.dot(psiN.T,meas))**2

	# prob1 = np.add.reduceat(probs, np.arange(0, len(probs), (N*M-1)))
	# prob1 = np.reshape(prob1, (M,N))

	# # complicated way to find entries with same state for particle 2, then add them up to find prob2, but it works

	# probs = np.reshape(probs, (N*M, N*M - 1))

	# for n in range(N*M):
	# 	for q in range(N*M - 1):
	# 		if q < n:
	# 			prob2[n] += probs[q, n-1]
	# 		if q > n:
	# 			prob2[n] += probs[q,n]
	# 		else:
	# 			prob2[n] += 0

	# prob2 = np.reshape(prob2, (M,N))


	# # save frames

	# fig = plt.figure(figsize=(6,3))

	# fig.suptitle('Reflective Edges', fontsize=8)

	# ax1 = fig.add_subplot(211)
	# col1 = ax1.pcolor(prob1, norm=LogNorm(vmin=0.001, vmax=1.0))
	# cbar1 = fig.colorbar(col1)
	# ax1.tick_params(labelsize=7)
	# cbar1.ax.tick_params(labelsize=7)
	# plt.title('particle 1 - probability', fontsize = 8)

	# ax2 = fig.add_subplot(212)
	# col2 = ax2.pcolor(prob2, norm=LogNorm(vmin=0.001, vmax=1.0))
	# cbar2 = fig.colorbar(col2)
	# ax2.tick_params(labelsize=7)
	# cbar2.ax.tick_params(labelsize=7)
	# plt.title('particle 2 - probability', fontsize = 8)

	# plt.subplots_adjust(hspace=0.4)

	# plt.savefig('2dwalkvidVarEdges'+str(r)+'.jpg', dpi=200)

	# plt.close(fig)

	# #######


	# MEAN DISTANCE

	weights = abs(psiN**2)
	weightedDist = weights*dist

	meanDist[r] = sum(weightedDist)/(sum(weights))

	steps += 1

	runningAvg[r] = (sum(meanDist[:steps]))/steps

	#######

finalDist = np.average(runningAvg[-5:])
print('final distance: ', finalDist)
print('total probability: ', sum(weights))



# PLOTTING

xAx = np.arange(stepsTot)

fig = plt.figure(figsize=(10,5), dpi=200)

# fig.suptitle('2 particles evolution - reflective edges - shape of graph: '+str(N)+' x '+ str(M), fontsize=8)

# ax1 = fig.add_subplot(211)
# plt.plot(xAx, meanDist)
# plt.xlabel('steps', fontsize=14, labelpad=1)
# plt.ylabel('inst distance', fontsize=14)
# ax1.tick_params(labelsize=14)

ax2 = fig.add_subplot(111)
plt.plot(xAx, runningAvg)
plt.xlabel('steps', fontsize=24, labelpad=1)
plt.ylabel('mean distance', fontsize=24)
ax2.tick_params(labelsize=24)

plt.subplots_adjust(hspace=0.0)

plt.show()