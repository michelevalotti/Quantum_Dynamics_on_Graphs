import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import scipy.linalg
from tqdm import tqdm # progress bar



def SquareTube(X,Y,gamma=1.0,structure='tube'):

    Adj = np.zeros((Y*X, Y*X))#, dtype=complex) # cannot hop diagonally, enumerate 0-steps**2 starting top left and ending bottom right (going left to right, starting at left)
    degree = np.eye((Y*X), dtype=complex)

    for i in range(X*Y): # i is y, j is x

        for j in range(X*Y):

            if abs(i-j) == 1 or abs(i-j) == Y:
                Adj[i,j] = 1
            if i%Y==0:
                Adj[i,i-1] = 0
            if j%Y==0:
                Adj[j-1,j] = 0

            if structure == 'tube':
                if j % Y == 0:
                    Adj[j,j+(Y-1)] = 1
                    Adj[j+(Y-1),j] = 1


    for i in range(Y*X):
        degree[i,i] = np.sum(Adj[i,:])


    H = gamma*(Adj-degree)

    return H


def ArrProbSq(H,X,Y,stepsTot,eta=1.0):

    EndProb = np.zeros(stepsTot)

    for v in range(X*Y):
        if v >= (Y*(X-1)): # add loss in all sites on the right (last column)
            H[v,v] -= 1j*(eta/2)

    for step in (range(stepsTot)):
        U = scipy.linalg.expm(-1j*H*step)
        psi0 = np.zeros((X*Y),dtype=complex)
        psi0[:Y] = 1 # initialise in a superposition of all nodes on the left
        psi0 = psi0/(np.sqrt(sum(psi0))) # normalise
        psiN = np.dot(U,psi0)

        probs = abs(psiN**2)

        EndProb[step] = 1.0-sum(probs)

    return EndProb, probs


def SDevSq(H,X,Y,stepsTot):

    SDev = np.zeros(stepsTot)
    xPosns = np.zeros(X*Y)
    for i in range(X*Y):
        xPosns[i] = int(i/Y)

    for step in (range(stepsTot)):
        U = scipy.linalg.expm(-1j*H*step)
        psi0 = np.zeros(X*Y)
        psi0[((int(X/2))*Y):(((int(X/2)+1)*Y))] = 1 # initialise in a superposition of states in the middle
        psi0 = psi0/(np.sqrt(sum(psi0))) # normalise
        psiN = np.dot(U,psi0)

        probs = abs(psiN**2)

        AvgX = sum(xPosns*probs)
        AvgXsq = sum((xPosns**2)*probs)

        StandDev = np.sqrt(np.around((AvgXsq - (AvgX)**2),decimals=10)) # if not rounded the first value in the sqrt is negative (order e-16)
        SDev[step] = StandDev

    return SDev, probs
    


if __name__ == '__main__':

    X = 20 # x size of side of lattice, keep higher than Y (horizontal tube)
    Y = 2 # y size of lattice
    Ylabel = Y
    gamma = 1.0 # hopping rate
    stepsTot = 100 # steps quantum particles takes on lattice
    stepsTotSD = int(stepsTot/4)
    eta = 1.0 # loss rate
    trialsTot = 6 # increase width of tube by one square per trial

    SDevAll = []
    ArrProbAll = []

    for t in tqdm(range(trialsTot)):
        H = SquareTube(X,Y)

        SDev,probsSD = SDevSq(H,X,Y,stepsTotSD)
        PlotProbsSD = np.zeros((Y,X))
        for i in range(X*Y):
            PlotProbsSD[i%Y][int(i/Y)] = probsSD[i]

        ArrProb, probsAP = ArrProbSq(H,X,Y,stepsTot)
        PlotProbsAP = np.zeros((Y,X))
        for i in range(X*Y):
            PlotProbsAP[i%Y][int(i/Y)] = probsAP[i]

        SDevAll.append(SDev)
        ArrProbAll.append(ArrProb)

        Y += 1


    # plot

    fig = plt.figure()
    gs1 = gridspec.GridSpec(3, 1)
    gs1.update(hspace=0.3)

    ax1 = fig.add_subplot(gs1[0,0])
    col1 = ax1.pcolor(PlotProbsAP)
    cbar1 = fig.colorbar(col1, label='probability')


    gs2 = gridspec.GridSpec(3, 1)
    gs2.update(hspace=0.3)

    ax2 = fig.add_subplot(gs2[1,0])
    for i in range(trialsTot):
        plt.plot(np.arange(stepsTot),ArrProbAll[i],label=('width: '+str(Ylabel+i)))
    plt.ylabel('Arrival Probability')
    plt.xlabel('steps')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax3 = fig.add_subplot(gs2[2,0])
    for j in range(trialsTot):
        plt.plot(np.arange(stepsTotSD),SDevAll[j],label=('width: '+str(Ylabel+j)))
    plt.xlabel('steps')
    plt.ylabel('$\sigma_x$')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()