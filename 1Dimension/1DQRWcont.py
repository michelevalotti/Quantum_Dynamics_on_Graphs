import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from tqdm import tqdm # progress bar

N = 101 # length of line
gamma = 0.4 # hopping rate
steps = 1000
initialPosn = int(N/2)


totTrials = 1000

# classical random walk

posnCl = np.zeros(N)
posnCl[initialPosn] = 1
index = 0

for n in tqdm(range(totTrials)): # n random walks
    for i in range(71): # throw coin i times
        flip = np.random.randint(0,2)
        if flip == 0:
            index += 1
        if flip == 1:
            index -= 1

    posnCl[index+initialPosn] += 1
    index = 0

posnCl = posnCl/(totTrials)
posnCl = posnCl[1::2]


# quantum random walk - continuous

Adj = np.zeros((N,N))
degree = np.eye(N)

for i in range(N):
    for j in range(N):
        if abs(i-j) == 1:
            Adj[i,j] = 1
    degree[i,i] = np.sum(Adj[i,:])


H = gamma*(degree-Adj)

StandDevArr = np.zeros(steps)


for step in tqdm(range(steps)):
    U = scipy.linalg.expm(-1j*H*step)

    psi0 = np.zeros(N)
    psi0[initialPosn] = 1
    psiN = np.dot(U,psi0)

    probs = abs(psiN**2)

    AvgX = sum((np.arange(N)-50)*probs)
    AvgXsq = sum(((np.arange(N)-50)**2)*probs)

    StandDev = np.sqrt(AvgXsq - (AvgX)**2)
    StandDevArr[step] = StandDev


# measurement

probs = np.zeros(N)

for i in range(N):
    meas = np.zeros(N)
    meas[i] = 1
    probs[i] = abs(np.dot(psiN.T,meas))**2

xAx = np.arange(N)


# plot

fig = plt.figure(figsize=(13,5))

# plt.title('comparing classical and quantum random walks')

ax1 = fig.add_subplot(211)
plt.plot(xAx, probs, label='continuous quantum walk', color='r')
plt.plot(xAx, probs,  'o', markersize=3, color='#FFA339')
plt.plot(xAx[1::2], posnCl, label='classical random walk', color='b')
plt.xlabel('position')
plt.ylabel('probability')
plt.legend()

ax2 = fig.add_subplot(212)
plt.plot(np.arange(steps), StandDevArr)
plt.xlabel('steps')
plt.ylabel('Standard Deviation from center')

plt.show()