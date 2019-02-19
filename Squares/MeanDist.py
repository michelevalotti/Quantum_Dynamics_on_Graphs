import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

N = 6 # size of side of lattice
gamma = 1 # hopping rate
steps = 0
init1 = 14 # initial position of part 1 (bottom left is 0, grows to the right and up, starts at left of every new row)
init2 = 20 # initial posn of part2
initialPosn = (init1*(N**2 - 1)) + (init2) # element of initial vector with value one
stepsTot = 100 # steps quantum particles takes on lattice
eta = 1.0
LossSite = 15


Adj = np.zeros((N**2,N**2), dtype=complex) # cannot hop diagonally, enumerate 0-steps**2 starting top left and ending bottom right (going left to right, starting at left each row)
degree = np.eye((N**4), dtype=complex)

for i in range(N**2): # i is y, j is x

	for j in range(N**2):

		if j == i-1 and (i%N!=0):
			Adj[i,j] = 1
		if j == i+1 and (j%N!=0):
			Adj[i,j] = 1
		if j == i - N:
			Adj[i,j] = 1
		if j == i + N:
			Adj[i,j] = 1


Adj = np.kron(Adj,Adj)

for i in range(N**4):
	degree[i,i] = np.sum(Adj[i,:])


for k in range(N**2-1,-1,-1):
	Adj = np.delete(Adj, (k*(N**2+1)), axis=0)
	Adj = np.delete(Adj, (k*(N**2+1)), axis=1)
	degree = np.delete(degree, (k*(N**2+1)), axis=0)
	degree = np.delete(degree, (k*(N**2+1)), axis=1)



# calculate distance between particles in each state
# and build dictionary of state to positions of particles

dist = np.zeros((N**4) - (N**2)) # array to store distances between particles in each two-particle state
positions = np.zeros(((N**4) - (N**2), 2)) # vocabulary of positons of particles for each state

for p in range(len(dist)):
	posn1 = int(p/(N**2 - 1))
	posn2 = p%(N**2 - 1)
	if posn2 >= posn1:
		posn2 = posn2 + 1

	positions[p] = np.array([posn1,posn2])

	x1 = posn1%N
	y1 = int(posn1/N)
	x2 = posn2%N
	y2 = int(posn2/N)
	dist[p] = np.sqrt((x1-x2)**2 + (y1-y2)**2)



# PARTICLES REPEL EACH OTHER

# divide distance in 3 sectors, the probability of staying in a lower sector is lower than staying
# in higher one,  highest sector has highest probability, probability of jumping between sectors is high.
# Lowest probability when two states swap.

maxDist = np.sqrt(2*((N-1)**2))
DistSectors = [maxDist/3, (2*maxDist)/3, maxDist]

ProbMatr = np.zeros(((N**4) - (N**2),(N**4) - (N**2)))

for n in range((N**4) - (N**2)):
	for m in range((N**4) - (N**2)):
		if ((dist[m] <= DistSectors[0]) and (dist[n] <= DistSectors[0])):
			ProbMatr[m,n] = 0.2
		elif (dist[m] < DistSectors[0] and dist[n] >= DistSectors[0]) or (dist[m] >= DistSectors[0] and dist[n] < DistSectors[0]):
			ProbMatr[m,n] = 0.9
		elif (((dist[m] > DistSectors[0]) and (dist[n] > DistSectors[0])) and ((dist[m] <= DistSectors[1]) and (dist[n] <= DistSectors[1]))):
			ProbMatr[m,n] = 0.4
		elif (dist[m] < DistSectors[1] and dist[n] >= DistSectors[1]) or (dist[m] >= DistSectors[1] and dist[n] < DistSectors[1]):
			ProbMatr[m,n] = 0.9
		elif (((dist[m] > DistSectors[1]) and (dist[n] > DistSectors[1])) and ((dist[m] <= DistSectors[2]) and (dist[n] <= DistSectors[2]))):
			ProbMatr[m,n] = 1.0
		elif (positions[m][0] == positions[n][1]) and (positions[m][1] == positions[n][0]): # particles exchange position
			ProbMatr[m,n] = 0.1


# check if matrix is symmetric

if np.allclose(ProbMatr, ProbMatr.T):
	print('symm')


# sub different probabilities in adjacency matrix only for enrties == 1 (allowed hops)

for i in range((N**4) - (N**2)):
	for j in range((N**4) - (N**2)):
		if Adj[i,j] == 1:
			Adj[i,j] = ProbMatr[i,j]

######


H = gamma*(degree-Adj)



# SIMPLE MODEL OF LOSS

# one site where there is prob particle 1 leaves the system, if it does, part2 also leaves
# e.g. diatomic molecule bond breaks and atoms fly out

for i in range(len(positions)):
	if positions[i][0] == LossSite:
		H[i,i] -= 1j*(eta/2)


######



meanDist = np.zeros(stepsTot)
runningAvg = np.zeros(stepsTot)

for r in range(stepsTot):


	U = scipy.linalg.expm(-1j*H*steps)

	psi0 = np.zeros((N**4)-(N**2))
	psi0[initialPosn] = 1
	psiN = np.dot(U,psi0)

	weights = abs(psiN**2)
	weightedDist = weights*dist

	meanDist[r] = sum(weightedDist)/(sum(weights))

	steps += 1

	runningAvg[r] = (sum(meanDist[:steps]))/steps

	print(sum(weights))
	print(steps)




# PLOTTING

xAx = np.arange(stepsTot)

fig = plt.figure()

fig.suptitle('2 particles evolution with repulsion and loss - reflective edges', fontsize=8)

ax1 = fig.add_subplot(211)
plt.plot(xAx, meanDist)
plt.xlabel('steps', fontsize=8, labelpad=1)
plt.ylabel('instantaneous mean distance', fontsize=7)
ax1.tick_params(labelsize=7)

ax2 = fig.add_subplot(212)
plt.plot(xAx, runningAvg)
plt.xlabel('steps', fontsize=8, labelpad=1)
plt.ylabel('mean distance - running avg', fontsize=7)
ax2.tick_params(labelsize=7)

plt.subplots_adjust(hspace=0.3)

plt.show()