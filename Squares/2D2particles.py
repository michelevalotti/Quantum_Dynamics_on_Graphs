import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.linalg

N = 4 # size of side of lattice
gamma = 1 # hopping rate
steps = 0
init1 = 14 # initial position of part 1 (bottom left is 0, grows to the right and up, starts at left of every new row)
init2 = 21 # initial posn of part2
initialPosn = (init1*(N**2 - 1)) + (init2) # element of initial vector with value one
stepsTot = 100 # steps quantum particles takes on lattice
eta = 0.5
LossSite = 15


Adj = np.zeros((N**2,N**2), dtype=complex) # cannot hop diagonally, enumerate 0-steps**2 starting top left and ending bottom right (going left to right, starting at left)
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

		# # boundaries loop around (torus)
		# if j == i + (N-1) and i%N==0:
		# 	Adj[i,j] = 1
		# if j == i + (N**2-N) and i<N:
		# 	Adj[i,j] = 1
		# if j == i - (N-1) and j%N==0:
		# 	Adj[i,j] = 1
		# if j == i - (N**2-N) and j<N:
		# 	Adj[i,j] = 1


Adj = np.kron(Adj,Adj)


for i in range(N**4):
	degree[i,i] = np.sum(Adj[i,:])


for k in range(N**2-1,-1,-1):
	Adj = np.delete(Adj, (k*(N**2+1)), axis=0)
	Adj = np.delete(Adj, (k*(N**2+1)), axis=1)
	degree = np.delete(degree, (k*(N**2+1)), axis=0)
	degree = np.delete(degree, (k*(N**2+1)), axis=1)


# REFLECTIVE EDGES


# calculate distance between particles in each state
# and build dictionary of state to positions of particles

dist = np.zeros((N**4) - (N**2)) # array to store distances between particles in each two-particle state
positions = np.zeros(((N**4) - (N**2), 2)) # dictionary of positons of particles for each state

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


if np.allclose(ProbMatr, ProbMatr.T):
	print('symm')

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


for r in range(stepsTot):


	U = scipy.linalg.expm(-1j*H*steps)

	psi0 = np.zeros((N**4)-(N**2))
	psi0[initialPosn] = 1
	psiN = np.dot(U,psi0)

	probs = np.zeros((N**4)-(N**2))
	prob2 = np.zeros(N**2)


	for m in range((N**4)-(N**2)):

		meas = np.zeros((N**4)-(N**2))
		meas[m] = 1
		probs[m] = abs(np.dot(psiN.T,meas))**2

	print(sum(probs))	

	prob1 = np.add.reduceat(probs, np.arange(0, len(probs), (N**2-1)))

	print('prob1: ', sum(prob1))

	prob1 = np.reshape(prob1, (N,N))


	# complicated way to find entries with same state for particle 2, then add them up to find prob2, but it works

	probs = np.reshape(probs, (N**2, N**2 - 1))

	for n in range(N**2):
		for q in range(N**2 - 1):
			if q < n:
				prob2[n] += probs[q, n-1]
			if q > n:
				prob2[n] += probs[q,n]
			else:
				prob2[n] += 0

	print('prob2: ',sum(prob2))

	prob2 = np.reshape(prob2, (N,N))
		 
	

	# fig = plt.figure(figsize=(6,2))

	# fig.suptitle('Reflective Edges', fontsize=8)


	# ax1 = fig.add_subplot(121)
	# col1 = ax1.pcolor(prob1, norm=LogNorm(vmin=0.001, vmax=1.0))
	# cbar1 = fig.colorbar(col1)
	# ax1.tick_params(labelsize=7)
	# cbar1.ax.tick_params(labelsize=7)
	# plt.title('particle 1 - probability', fontsize = 8)

	# ax2 = fig.add_subplot(122)
	# col2 = ax2.pcolor(prob2, norm=LogNorm(vmin=0.001, vmax=1.0))
	# cbar2 = fig.colorbar(col2)
	# ax2.tick_params(labelsize=7)
	# cbar2.ax.tick_params(labelsize=7)
	# plt.title('particle 2 - probability', fontsize = 8)

	# plt.subplots_adjust(top=0.8)

	# plt.savefig('2dwalkvidRepulsionLoss'+str(r)+'.jpg', dpi=200)

	# plt.close(fig)

	steps += 1
	print(steps)