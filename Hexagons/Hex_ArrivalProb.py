import matplotlib.pyplot as plt
import numpy as np
from Hex_1part import ArrivalProbability


HopRate = 1.0
Steps = 200
LossRate = 1.0 # loss rate -- keep high
trials = 5

X = 6 # x - horizontal cylinder (M > N) only works if this is even
Y = 4 # y
ChiralShift = 0

StepsToArrivalHor_length = np.zeros(trials)
LenOfTubeHor_length = np.zeros(trials)
StepsToArrivalVert_length = np.zeros(trials)
LenOfTubeVert_length = np.zeros(trials)
StepsToArrivalChiral_length = np.zeros(trials)
LenOfTubeChiral_length = np.zeros(trials)


# calculate how many steps it takes to reach 50% arrival probability for tubes of fixed widht and increasing length

for i in range(2): # horizontal first, vertical second, chiral third
    if X > Y:
        if ChiralShift != 0:
            X = 6
            Y = 4
            print('chiral length')
            for i in range(trials):
                myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate,ChiralShift)
                EndProb = myValues[5]
                for j in range(len(EndProb)):
                    if abs(EndProb[j] - 0.5) < 0.01:
                        # print(EndProb[j])
                        StepsToArrivalChiral_length[i] = j
                LenOfTubeChiral_length[i] = X
                X += 6

            X = 4
            Y = 6 

        else:
            print('horizontal length')
            for i in range(trials):
                myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate,ChiralShift)
                EndProb = myValues[5]
                for j in range(len(EndProb)):
                    if abs(EndProb[j] - 0.5) < 0.01:
                        # print(EndProb[j])
                        StepsToArrivalHor_length[i] = j
                LenOfTubeHor_length[i] = X
                X += 6 # keeps X even
        
    
    if X < Y:
        print('vertical length')
        ChiralShift = 0
        for i in range(trials):
            myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate,ChiralShift)
            EndProb = myValues[5]
            for j in range(len(EndProb)):
                if abs(EndProb[j] - 0.5) < 0.01:
                    # print(EndProb[j])
                    StepsToArrivalVert_length[i] = j
            LenOfTubeVert_length[i] = Y
            Y += 6 # keep even

    ChiralShift = 1



# calculate how many steps it takes to reach 50% arrival probability for tubes of fixed length and increasing width

# reinitialise values
X = 20
Y = 2
ChiralShift = 0

StepsToArrivalHor_width = np.zeros(trials)
LenOfTubeHor_width = np.zeros(trials)
StepsToArrivalVert_width = np.zeros(trials)
LenOfTubeVert_width = np.zeros(trials)
StepsToArrivalChiral_width = np.zeros(trials)
LenOfTubeChiral_width = np.zeros(trials)

for i in range(2): # horizontal first and vertical second
    if X > Y:
        if ChiralShift != 0:
            X = 20
            Y = 2
            print('chiral width')
            for i in range(trials):
                myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate,ChiralShift)
                EndProb = myValues[5]
                for j in range(len(EndProb)):
                    if abs(EndProb[j] - 0.5) < 0.01:
                        # print(EndProb[j])
                        StepsToArrivalChiral_width[i] = j
                LenOfTubeChiral_width[i] = Y
                Y += 1

            X = 2
            Y = 20
    
        else:
            print('horizontal width')
            for i in range(trials):
                myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate,ChiralShift)
                EndProb = myValues[5]
                for j in range(len(EndProb)):
                    if abs(EndProb[j] - 0.5) < 0.01:
                        # print(EndProb[j])
                        StepsToArrivalHor_width[i] = j
                LenOfTubeHor_width[i] = Y
                Y += 1


    
    if X < Y:
        print('vertical width')
        for i in range(trials):
            myValues = ArrivalProbability(Y,X,HopRate,Steps,LossRate,ChiralShift)
            EndProb = myValues[5]
            for j in range(len(EndProb)):
                if abs(EndProb[j] - 0.5) < 0.01:
                    # print(EndProb[j])
                    StepsToArrivalVert_width[i] = j
            LenOfTubeVert_width[i] = X
            X += 1

    ChiralShift = 1



fig = plt.figure()

ax1 = fig.add_subplot(121)
plt.plot(LenOfTubeHor_length, StepsToArrivalHor_length, 'o')
plt.plot(LenOfTubeHor_length, StepsToArrivalHor_length, label='zigzag', color='b')
plt.plot(LenOfTubeVert_length, StepsToArrivalVert_length, 'o')
plt.plot(LenOfTubeVert_length, StepsToArrivalVert_length, label='armchair', color='r')
plt.plot(LenOfTubeChiral_length, StepsToArrivalChiral_length, 'o')
plt.plot(LenOfTubeChiral_length, StepsToArrivalChiral_length, label='chiral', color='g')
plt.legend()
plt.xlabel('length of tube')
plt.ylabel('steps')

ax2 = fig.add_subplot(122)
plt.plot(LenOfTubeHor_width, StepsToArrivalHor_width, 'o')
plt.plot(LenOfTubeHor_width, StepsToArrivalHor_width, label='zigzag', color='b')
plt.plot(LenOfTubeVert_width, StepsToArrivalVert_width, 'o')
plt.plot(LenOfTubeVert_width, StepsToArrivalVert_width, label='armchair', color='r')
plt.plot(LenOfTubeChiral_width, StepsToArrivalChiral_width, 'o')
plt.plot(LenOfTubeChiral_width, StepsToArrivalChiral_width, label='chiral', color='g')
plt.legend()
plt.xlabel('width of tube')
plt.ylabel('steps')

plt.show()