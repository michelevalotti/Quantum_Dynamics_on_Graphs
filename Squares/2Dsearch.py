import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

N = 21 # size of side of lattice (best if odd)
gamma = 0.1 # hopping rate (if == 1 gives very jagged plot, seems random)
markedPosn = 5 # marked state in (5,5) posn
steps = 0 # initialise steps quantum particles takes on lattice
xDim = 100 # dimesion of x axis (total number of steps)



Adj = np.zeros((N**2,N**2)) # cannot hop diagonally, enumerate 0-steps**2 starting top left and ending bottom right (going left to right, starting at left)
degree = np.eye(N**2)


for i in range(N**2):

	for j in range(N**2):

		if j == i-1 and (i%N!=0):
			Adj[i,j] = 1
		if j == i+1 and (j%N!=0):
			Adj[i,j] = 1
		if j == i - N:
			Adj[i,j] = 1
		if j == i + N:
			Adj[i,j] = 1

	degree[i,i] = np.sum(Adj[i,:])



AdjTorus = np.zeros((N**2,N**2)) # cannot hop diagonally, enumerate 0-steps**2 starting top left and ending bottom right (going left to right, starting at left)
degreeTorus = np.eye(N**2)

for i in range(N**2):

	for j in range(N**2):

		if j == i-1 and (i%N!=0):
			AdjTorus[i,j] = 1
		if j == i+1 and (j%N!=0):
			AdjTorus[i,j] = 1
		if j == i - N:
			AdjTorus[i,j] = 1
		if j == i + N:
			AdjTorus[i,j] = 1

		# boundaries loop around (torus)
		if j == i + (N-1) and i%N==0:
			AdjTorus[i,j] = 1
		if j == i + (N**2-N) and i<N:
			AdjTorus[i,j] = 1
		if j == i - (N-1) and j%N==0:
			AdjTorus[i,j] = 1
		if j == i - (N**2-N) and j<N:
			AdjTorus[i,j] = 1

	degreeTorus[i,i] = np.sum(Adj[i,:])

probs = np.zeros(xDim)
probsTorus = np.zeros(xDim)

for i in range(xDim):

	posn = np.zeros(N)
	marked = np.copy(posn)
	marked[markedPosn] = 1
	marked = np.kron(marked,marked)

	test = np.copy(posn)
	test[20] = 1
	test = np.kron(test,test)


	H = gamma*(degree-Adj)
	H[(markedPosn*N + markedPosn),(markedPosn*N + markedPosn)] = -H[(markedPosn*N + markedPosn),(markedPosn*N + markedPosn)]
	U = scipy.linalg.expm(-1j*H*steps)

	Htorus = gamma*(degreeTorus-AdjTorus)
	Htorus[(markedPosn*N + markedPosn),(markedPosn*N + markedPosn)] = -Htorus[(markedPosn*N + markedPosn),(markedPosn*N + markedPosn)]
	Utorus = scipy.linalg.expm(-1j*Htorus*steps)


	psi0 = np.kron(posn,posn)
	psi0 += (1/N) # particle in a superposition of all states
	psiN = np.dot(U,psi0)

	psiNtorus = np.dot(Utorus,psi0)

	probs[i] = abs(np.dot(psiN.T,marked))**2
	probsTorus[i] = abs(np.dot(psiNtorus.T,marked))**2


	steps += 1
	print(steps)


xAx = np.arange(xDim)


fig = plt.figure()

ax1 = fig.add_subplot(211)

plt.plot(xAx, probs)
plt.xlabel('steps')
plt.ylabel('P(marked)')
# plt.title('Continuous QRW reflective edges -- '+str(xDim)+' steps and hopping rate = '+str(gamma))


ax2 = fig.add_subplot(212)

plt.plot(xAx, probsTorus)
plt.xlabel('steps')
plt.ylabel('P(marked)')
# plt.title('Continuous QRW Torus -- '+str(xDim)+' steps and hopping rate = '+str(gamma))

plt.subplots_adjust(hspace=0.0)

plt.show()