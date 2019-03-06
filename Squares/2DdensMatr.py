import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import scipy.linalg
from tqdm import tqdm # progress bar

N = 17 # size of side of lattice #19
gamma = 1.0 # hopping rate
steps = 10 # steps quantum particles takes on lattice #11
initialPosn = 8 #9
decRate = 0.2 # decoherence rate -- starts to break normalization (decreasing probability with more steps, or higher than 1 if rate > 1)

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


H = gamma*(degree-Adj)

# # add potential
# H[125,125] += H[125,125]
# H[145,145] -= H[145,145]

# add potential gradient
for i in range(N**2):
	H[i,i] -= (i%N)/3

posn = np.zeros(N)
posn[initialPosn] = 1
psi0 = np.kron(posn,posn)
# psi0 += 1/(float(N))
psiN = np.outer(psi0,psi0)
step = 0

for k in tqdm(range(steps)):

	U = scipy.linalg.expm(-1j*H*step)

	psiN = np.dot(U,psiN)
	psiN = np.dot(psiN,U.T)

	# decoherence

	decOp = np.zeros((N**2,N**2), dtype=complex)

	for j in range(N**2):
		proj = np.zeros(N**2)
		proj[j] = 1
		projMatr = np.outer(proj,proj)
		decoherence = np.dot(projMatr, psiN)
		decoherence = np.dot(decoherence, projMatr.T)
		decOp += decoherence

	psiN = (1-decRate) * psiN

	psiN += decRate*decOp

	step += 0.06


probs = np.zeros(N**2)

for i in range(N**2):
	
	meas = np.zeros(N**2)
	meas[i] = 1
	RhoMeas = np.outer(meas,meas.T)
	probs[i] = np.absolute(np.trace(np.dot(psiN,RhoMeas)))

print(sum(probs), 'ProbTot Refl')

probs = np.reshape(probs, (N,N))

probsX = probs.sum(axis=0)
probsY = probs.sum(axis=1)

xAx = np.arange(N)


# # debug

# Hvals = np.zeros(N**2)
# for m in range(N**2):
# 	Hvals[m] = H[m,m]

# Hvals = np.reshape(Hvals,(N,N))

# print(Hvals.shape)


# # debug plot

# fig = plt.figure()
# ax1 = plt.subplot(111)
# col1 = ax1.pcolor(Hvals,)
# cbar = fig.colorbar(col1)
# plt.show()



# plot

fig = plt.figure(figsize=(6,5), dpi=200)

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])

# fig.suptitle('Number of steps = '+str(steps)+' and Decoherence rate = '+str(decRate))

ax1 = plt.subplot(gs[1])
col1 = ax1.pcolor(probs)#, norm=LogNorm(vmin=0.001, vmax=1.0))
plt.axis('off')


ax2 = plt.subplot(gs[0])
plt.plot(probsY, xAx)
plt.xlabel('$P_{tot}$', fontsize=14)
plt.ylabel('y', rotation=0, fontsize=14)
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top') 
ax2.yaxis.labelpad = 10
ax2.tick_params(labelsize=14)


ax3 = plt.subplot(gs[3])
plt.plot(xAx, probsX)
plt.ylabel('$P_{tot}$', rotation=0, fontsize=14)
plt.xlabel('x', fontsize=14)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position('right')
ax3.yaxis.labelpad = 12
ax3.tick_params(labelsize=14)

plt.subplots_adjust(hspace=0.0, wspace=0.0)
plt.subplots_adjust(left=0.15, right=0.8)
plt.show()