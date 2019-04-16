import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.linalg
from tqdm import tqdm # progress bar

N = 121 # length of line
gamma = 0.5 # hopping rate
steps = 40
initialPosn = int(N/2)
decRates = [0.0, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 1.0] # decoherence rates


Adj = np.zeros((N,N))
degree = np.eye(N)

for i in range(N):
	for j in range(N):
		if abs(i-j) == 1:
			Adj[i,j] = 1
	degree[i,i] = np.sum(Adj[i,:])


H = gamma*(degree-Adj)

psi0 = np.zeros(N, dtype=complex)
psi0[initialPosn] = 1

probs = np.zeros((len(decRates), N))

for m in tqdm(range(len(decRates))):

	psiN = np.outer(psi0,psi0)
	step = 0
	decRate = decRates[m]

	for k in tqdm(range(steps)):

		U = scipy.linalg.expm(-1j*H*step)

		psiN = np.dot(U,psiN)
		psiN = np.dot(psiN,U.T)

		# decoherence

		decOp = np.zeros((N,N), dtype=complex)

		for j in range(N):
			proj = np.zeros(N)
			proj[j] = 1
			projMatr = np.outer(proj,proj)
			decoherence = np.dot(projMatr, psiN)
			decoherence = np.dot(decoherence, projMatr.T)
			decOp += decoherence

		psiN = (1-decRate) * psiN

		psiN += decRate*decOp

		step += 0.06


	for i in range(N):
		
		meas = np.zeros(N)
		meas[i] = 1
		RhoMeas = np.outer(meas,meas.T)
		probs[m][i] = np.absolute(np.trace(np.dot(psiN,RhoMeas)))
	print(sum(probs[m]))

xAx = np.arange(N)


# plot

fig = plt.figure(figsize=(14,5), dpi=200)

# plt.title('QRW and decoherence')

ax1 = fig.add_subplot(111)
for m in range(len(decRates)):
	plt.plot(xAx, probs[m], label='p = '+ str(decRates[m]))
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=14)
plt.xlabel('position', fontsize=14)
plt.ylabel('probability distribution', fontsize=14)
ax1.tick_params(labelsize=14)

plt.show()