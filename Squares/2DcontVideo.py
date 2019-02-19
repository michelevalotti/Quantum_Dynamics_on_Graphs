import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.linalg

N = 21 # size of side of lattice
gamma = 0.2 # hopping rate
steps = 0 # steps quantum particles takes on lattice
markedXposn = 5 # X position of marked state
markedYposn = 8 # Y position of marked state
initialPosn = 10 # initial state position (10,10), not for search
stepsTot = 100
eta = 0.9 # loss coefficient



Zero = np.array([1,0]) # ket 0
One = np.array([0,1]) # ket 1


Adj = np.zeros((N**2,N**2), dtype=complex) # cannot hop diagonally, enumerate 0-steps**2 starting top left and ending bottom right (going left to right, starting at left)
degree = np.eye((N**2), dtype=complex)


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



AdjTorus = np.zeros((N**2,N**2), dtype=complex) # cannot hop diagonally, enumerate 0-steps**2 starting top left and ending bottom right (going left to right, starting at left)
degreeTorus = np.eye((N**2), dtype=complex)

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

H = gamma*(degree-Adj)
Htorus = gamma*(degreeTorus-AdjTorus)

# add loss -- particle can escape lattice in state i (H[i,i])with probability eta
H[22,22] -= 1j*(eta/2)
Htorus[22,22] -= 1j*(eta/2)


# # add potential (+ve) or marked state (-ve)
# H[(markedYposn*N + markedXposn),(markedYposn*N + markedXposn)] = -H[(markedYposn*N + markedXposn),(markedYposn*N + markedXposn)]
# Htorus[(markedYposn*N + markedXposn),(markedYposn*N + markedXposn)] = -Htorus[(markedYposn*N + markedXposn),(markedYposn*N + markedXposn)]
# # H[95,95] += 8
# Htorus[95,95] += 8
# H[74,74] += 8
# Htorus[74,74] += 8
# H[73,73] += 8
# Htorus[73,73] += 8


probsMarked = []
probsMarkedTorus = []


for k in range(stepsTot):
	
	posn = np.zeros(N)

	U = scipy.linalg.expm(-1j*H*steps)

	Utorus = scipy.linalg.expm(-1j*Htorus*steps)


	# posn[np.ceil((N)/2)]=1 # particle starts in the middle
	posn[initialPosn] = 1 # particle starts in position (initailPosn, initialPosn), not for search
	psi0 = np.kron(posn,posn)
	# psi0 += (1/float(N)) # uniform superposition of all states, for search
	
	psiN = np.dot(U,psi0)
	
	psiNtorus = np.dot(Utorus,psi0)


	# markedX = np.copy(posn)
	# markedY = np.copy(posn)
	# markedX[markedXposn] = 1
	# markedY[markedYposn] = 1
	# marked = np.kron(markedY,markedX)


	probs = np.zeros(N**2)

	# probsMarked.append(abs(np.dot(psiN.T,marked))**2)


	for i in range(N**2):

		meas = np.zeros(N**2)
		meas[i] = 1
		probs[i] = abs(np.dot(psiN.T,meas))**2

	print(sum(probs), 'ProbTot Refl')

	probs = np.reshape(probs, (N,N))



	probsTorus = np.zeros(N**2)

	# probsMarkedTorus.append(abs(np.dot(psiNtorus.T,marked))**2)


	for i in range(N**2):

		meas = np.zeros(N**2)
		meas[i] = 1
		probsTorus[i] = abs(np.dot(psiNtorus.T,meas))**2

	print(sum(probsTorus), 'ProbTot Torus')

	probsTorus = np.reshape(probsTorus, (N,N))



	fig = plt.figure(figsize=(5,3))

	xAx = np.arange(k+1)

	# fig.suptitle('Quantum search, continuous walk - marked state in ('+str(markedXposn)+','+str(markedYposn)+')', fontsize=8) # title for search plot
	# fig.suptitle('Continuous quantum walk - initial state in ('+str(initialPosn)+','+str(initialPosn)+')', fontsize=10) # title for search plot
	fig.suptitle('Quantum walk with loss in state (1,1)', fontsize=8) # title for search plot


	ax1 = fig.add_subplot(211)

	col1 = ax1.pcolor(probs, norm=LogNorm(vmin=0.001, vmax=1.0))
	cbar1 = fig.colorbar(col1)
	# cbar1.set_ticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
	# cbar1.set_ticks([0.00, 0.003, 0.006, 0.009, 0.012, 0.015])
	# cbar1.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

	plt.title('Reflective edges', fontsize=7)
	ax1.tick_params(labelsize=7)
	cbar1.ax.tick_params(labelsize=7)

	plt.xlim(0, N)
	plt.ylim(0, N)


	# ax3 = fig.add_subplot(222)

	# plt.plot(xAx, probsMarked)
	# plt.xlabel('steps', fontsize=7, labelpad=1)
	# plt.ylabel('P(marked)', fontsize=7)

	# ax3.tick_params(labelsize=7)
	# ax3.yaxis.set_label_position("right")

	# plt.xlim(0, stepsTot)
	# plt.ylim(0, 0.015)



	ax2 = fig.add_subplot(212)

	col2 = ax2.pcolor(probsTorus, norm=LogNorm(vmin=0.001, vmax=1.0))
	cbar2 = fig.colorbar(col2)
	# cba2.set_ticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
	# cbar2.set_ticks([0.00, 0.005, 0.01, 0.015, 0.02])
	# cbar1.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

	plt.title('Torus', fontsize=7)
	ax2.tick_params(labelsize=7)
	cbar2.ax.tick_params(labelsize=7)
	
	plt.xlim(0, N)
	plt.ylim(0, N)



	# ax4 = fig.add_subplot(224)

	# plt.plot(xAx, probsMarkedTorus)
	# plt.xlabel('steps', fontsize=7, labelpad=1)
	# plt.ylabel('P(marked)', fontsize=7)

	# ax4.tick_params(labelsize=7)
	# ax4.yaxis.set_label_position("right")

	# plt.xlim(0, stepsTot)
	# plt.ylim(0, 0.02)


	# plt.subplots_adjust(hspace=0.5, top=0.85, wspace=0.4)
	plt.subplots_adjust(hspace=0.5)


	plt.savefig('2dwalkvidLoss'+str(k)+'.jpg', dpi=200)

	plt.close(fig)

	steps += 1
	print(steps)