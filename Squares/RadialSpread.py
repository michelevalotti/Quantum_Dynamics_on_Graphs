import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import scipy.linalg
from tqdm import tqdm # progress bar
spreading = __import__('2Dspreading')


def SDevSqRad(H,X,Y,stepsTot):

    SDev = np.zeros(stepsTot)
    xPosns = np.zeros(X*Y)
    yPosns = np.zeros(X*Y)
    RadialDist = np.zeros(X*Y)
    middleX = int(X/2)
    middleY = int(Y/2)
    for i in range(X*Y):
        xPosns[i] = int(i/Y)
        yPosns[i] = i%Y
        RadialDist[i] = np.sqrt((abs(xPosns[i]-middleX)**2)+(abs(yPosns[i]-middleY)**2))

    for step in (range(stepsTot)):
        U = scipy.linalg.expm(-1j*H*step)
        psi0 = np.zeros(X*Y)
        if X%2 != 0:
            psi0[int((X*Y)/2)] = 1 # initialise in the middle
        else:
            psi0[int((X*Y)/2-Y/2)] = 1 # initialise in the middle
        psiN = np.dot(U,psi0)

        probs = abs(psiN**2)

        AvgX = sum(RadialDist*probs)
        AvgXsq = sum((RadialDist**2)*probs)

        StandDev = np.sqrt(np.around((AvgXsq - (AvgX)**2),decimals=10)) # if not rounded the first value in the sqrt is negative (order e-16)
        SDev[step] = StandDev

    return SDev, probs



if __name__ == "__main__":

    X = 10 # x size of side of lattice, keep higher than Y (horizontal tube)
    Y = 10 # y size of lattice
    Ylabel = Y
    stepsTot = 20
    eta = 1.0 # loss rate
    trialsTot = 6 # decrease width of tube by one square per trial

    SDevRadAll = []

    for t in tqdm(range(trialsTot)):
        H = spreading.SquareTube(X,Y,structure='lattice')
        
        SDevRad, probsRad = SDevSqRad(H,X,Y,stepsTot)
        if t == 0:
            PlotProbsRad = np.zeros((Y,X))
            for i in range(X*Y):
                PlotProbsRad[i%Y][int(i/Y)] = probsRad[i]

        SDevRadAll.append(SDevRad)

        Y -= 1


    # plot

    fig = plt.figure()
    gs1 = gridspec.GridSpec(6, 5)
    gs1.update(hspace=0.3)


    ax2 = fig.add_subplot(gs1[:4,1:4])
    col2 = ax2.pcolor(PlotProbsRad)
    cbar2 = fig.colorbar(col2, label='probability')


    gs2 = gridspec.GridSpec(6, 5)
    gs2.update(hspace=0.3)

    ax5 = fig.add_subplot(gs2[4:,:])
    for j in range(trialsTot):
        plt.plot(np.arange(stepsTot),SDevRadAll[j],label=('width: '+str(Ylabel-j)))
    plt.xlabel('steps')
    plt.ylabel('$\sigma_r$')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()