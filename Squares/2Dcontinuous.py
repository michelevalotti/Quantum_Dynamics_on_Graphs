import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

N = 31 # size of side of lattice (best if odd)
gamma = 0.3 # hopping rate
steps = 15 # steps quantum particles takes on lattice
potPsn = 67

Zero = np.array([1,0]) # ket 0
One = np.array([0,1]) # ket 1
posn = np.zeros(N)


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


H = gamma*(degree-Adj)

# # add potential to reflective edges
# H[potPsn,potPsn] += 8

U = scipy.linalg.expm(-1j*H*steps)

Htorus = gamma*(degreeTorus-AdjTorus)

# # add potential to torus
# Htorus[potPsn,potPsn] += 8

Utorus = scipy.linalg.expm(-1j*Htorus*steps)


posn[int((N)/2)]=1 # particle starts in the middle
psi0 = np.kron(posn,posn)
psiN = np.dot(U,psi0)

psiNtorus = np.dot(Utorus,psi0)

print(sum(psiN))
print(sum(psiNtorus))

probs = np.zeros(N**2)

for i in range(N**2):

	meas = np.zeros(N**2)
	meas[i] = 1
	probs[i] = abs(np.dot(psiN.T,meas))**2

probs = np.reshape(probs, (N,N))



probsTorus = np.zeros(N**2)

for i in range(N**2):

	meas = np.zeros(N**2)
	meas[i] = 1
	probsTorus[i] = abs(np.dot(psiNtorus.T,meas))**2

probsTorus = np.reshape(probsTorus, (N,N))



fig = plt.figure(figsize=(4,7), dpi=200)

ax1 = fig.add_subplot(211)

col1 = ax1.pcolor(probs)
cbar1 = fig.colorbar(col1)
cbar1.ax.tick_params(labelsize=14)

plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
ax1.tick_params(labelsize=14)
# plt.title('Continuous QRW reflective edges -- '+str(steps)+' steps and hopping rate = '+str(gamma))

plt.xlim(0, N)
plt.ylim(0, N)


ax2 = fig.add_subplot(212)

col2 = ax2.pcolor(probsTorus)
cbar2 = fig.colorbar(col2)
cbar2.ax.tick_params(labelsize=14)


plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
ax2.tick_params(labelsize=14)
# plt.title('Continuous QRW Torus -- '+str(steps)+' steps and hopping rate = '+str(gamma))

plt.xlim(0, N)
plt.ylim(0, N)

plt.subplots_adjust(hspace=0.5, left=0.15, right=0.89)

plt.show()