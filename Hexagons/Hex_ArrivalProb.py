import matplotlib.pyplot as plt
import numpy as np
from Hex_1part import ArrivalProbability

X = 6 # x - horizontal cylinder (M > N) only works if this is even
Y = 4 # y
HopRate = 1.0
Steps = 200
LossRate = 1.0 # loss rate -- keep high
trials = 7

StepsToArrivalHor_length = np.zeros(trials)
LenOfTubeHor_length = np.zeros(trials)
StepsToArrivalVert_length = np.zeros(trials)
LenOfTubeVert_length = np.zeros(trials)


# calculate how many steps it takes to reach 50% arrival probability for tubes of fixed widht and increasing length

for i in range(2): # horizontal first and vertical second
    if X > Y:
        for i in range(trials):
            myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate)
            EndProb = myValues[5]
            for j in range(len(EndProb)):
                if abs(EndProb[j] - 0.5) < 0.01:
                    # print(EndProb[j])
                    StepsToArrivalHor_length[i] = j
            LenOfTubeHor_length[i] = X
            X += 6
    if X < Y:
        for i in range(trials):
            myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate)
            EndProb = myValues[5]
            for j in range(len(EndProb)):
                if abs(EndProb[j] - 0.5) < 0.01:
                    # print(EndProb[j])
                    StepsToArrivalVert_length[i] = j
            LenOfTubeVert_length[i] = Y
            Y += 6

    X = 4
    Y = 6 


# calculate how many steps it takes to reach 50% arrival probability for tubes of fixed length and increasing width

# reinitialise X anad Y values (keep even)
X = 6
Y = 4

StepsToArrivalHor_width = np.zeros(trials)
LenOfTubeHor_width = np.zeros(trials)
StepsToArrivalVert_width = np.zeros(trials)
LenOfTubeVert_width = np.zeros(trials)

for i in range(2): # horizontal first and vertical second
    if X > Y:
        # reinitialise X anad Y values (keep even)
        X = 20
        Y = 2
        for i in range(trials):
            myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate)
            EndProb = myValues[5]
            for j in range(len(EndProb)):
                if abs(EndProb[j] - 0.5) < 0.01:
                    # print(EndProb[j])
                    StepsToArrivalHor_width[i] = j
            LenOfTubeHor_width[i] = Y
            Y += 1
    if X < Y:
        # reinitialise X anad Y values (keep even)
        X = 2
        Y = 20
        for i in range(trials):
            myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate)
            EndProb = myValues[5]
            for j in range(len(EndProb)):
                if abs(EndProb[j] - 0.5) < 0.01:
                    # print(EndProb[j])
                    StepsToArrivalVert_width[i] = j
            LenOfTubeVert_width[i] = X
            X += 1

    X = 4
    Y = 6

fig = plt.figure()

ax1 = fig.add_subplot(121)
plt.plot(LenOfTubeHor_length, StepsToArrivalHor_length, 'o')
plt.plot(LenOfTubeHor_length, StepsToArrivalHor_length, label='horizontal', color='b')
plt.plot(LenOfTubeVert_length, StepsToArrivalVert_length, 'o')
plt.plot(LenOfTubeVert_length, StepsToArrivalVert_length, label='vertical', color='r')
plt.legend()
plt.xlabel('length of tube')
plt.ylabel('steps')

ax2 = fig.add_subplot(122)
plt.plot(LenOfTubeHor_width, StepsToArrivalHor_width, 'o')
plt.plot(LenOfTubeHor_width, StepsToArrivalHor_width, label='horizontal', color='b')
plt.plot(LenOfTubeVert_width, StepsToArrivalVert_width, 'o')
plt.plot(LenOfTubeVert_width, StepsToArrivalVert_width, label='vertical', color='r')
plt.legend()
plt.xlabel('width of tube')
plt.ylabel('steps')

plt.show()