import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from tqdm import tqdm # progress bar

N = 101 # length of line
gamma = 0.4 # hopping rate
steps = 51 # keep odd for plotting classical 
initialPosn = int(N/2)


totTrials = 10000

# classical random walk

index = 0
SDevCl = np.zeros(steps)


for s in tqdm(range(steps)):
    posnCl = np.zeros(N)
    for n in (range(totTrials)): # n random walks
        ProbCl = np.zeros(N)
        for i in range(s): # throw coin i times
            flip = np.random.randint(0,2)
            if flip == 0:
                index += 1
            if flip == 1:
                index -= 1

        posnCl[index+initialPosn] += 1
        index = 0

    posnCl = posnCl/(totTrials)
    AvgX = sum(posnCl*(np.arange(N)))
    AvgXsq = sum(posnCl*((np.arange(N))**2))
    SDev = np.sqrt(AvgXsq - (AvgX)**2)
    SDevCl[s] += SDev


posnCl = posnCl[0::2] # only plot non zero values


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

    AvgX = sum((np.arange(N))*probs)
    AvgXsq = sum(((np.arange(N))**2)*probs)

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
plt.plot(xAx[0::2], posnCl, label='classical random walk', color='b') # [1::2]
plt.xlabel('position')
plt.ylabel('probability')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

ax2 = fig.add_subplot(212)
plt.plot(np.arange(steps), StandDevArr, color='r')
plt.plot(np.arange(steps), SDevCl, color='b')
plt.xlabel('steps')
plt.ylabel('$\sigma_x$')

plt.subplots_adjust(right=0.8)

plt.show()