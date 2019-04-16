from numpy import *
from matplotlib.pyplot import *
from tqdm import tqdm # progress bar

N = 50 # number of random steps
P = 2*N + 1 # number of positions 
tot_trials = 10000 # for classical random walk


# classical random walk

index = 0
SDevCl = np.zeros(N)


for s in tqdm(range(N)):
    posnCl = np.zeros(P)
    for n in (range(tot_trials)): # n random walks
        ProbCl = np.zeros(P)
        for i in range(s): # throw coin i times
            flip = np.random.randint(0,2)
            if flip == 0:
                index += 1
            if flip == 1:
                index -= 1

        posnCl[index+N] += 1
        index = 0

    posnCl = posnCl/(tot_trials)
    AvgX = sum(posnCl*(arange(P)))
    AvgXsq = sum(posnCl*((arange(P))**2))
    SDev = np.sqrt(AvgXsq - (AvgX)**2)
    SDevCl[s] += SDev


posnCl = posnCl[1::2] # only plot non zero values


# quantum random walk

coin0 = array([1,0]) # ket 0
coin1 = array([0,1]) # ket 1


# outer products

C00 = outer(coin0, coin0)
C01 = outer(coin0, coin1)
C10 = outer(coin1, coin0)
C11 = outer(coin1, coin1)


# coin operator to flip quantum coin

C_hat = (C00 + C01 + C10 - C11)/sqrt(2.0)


# shift (step) operator

ShiftPlus = roll(eye(P), 1, axis=0)
ShiftMinus = roll(eye(P), -1, axis=0)
S_hat = kron(ShiftPlus, C00) + kron(ShiftMinus, C11)


# walk operator

U = S_hat.dot(kron(eye(P), C_hat))

StandDevArr = np.zeros(N)

for n in tqdm(range(N)):
    posn0 = zeros(P)
    posn0[N] = 1 # N is central position
    psi0 = kron(posn0, (coin0+coin1*1j)/sqrt(2.0)) # symmetric initial coin (superposition of |0> and |1>)
    psiN = linalg.matrix_power(U, n).dot(psi0)

    # measurement
    prob = empty(P)
    for k in range(P):
        posn = zeros(P)
        posn[k] = 1
        M_hat_k = kron(outer(posn, posn), eye(2))
        proj = M_hat_k.dot(psiN)
        prob[k] = proj.dot(proj.conjugate()).real

    AvgX = sum((arange(P))*prob)
    AvgXsq = sum(((arange(P))**2)*prob)

    StandDev = np.sqrt(AvgXsq - (AvgX)**2)
    StandDevArr[n] = StandDev

    prob = prob[1::2]


# plot

fig = figure(figsize=(14,5), dpi=200)

# title('comparing classical and quantum random walks')

# ax1 = fig.add_subplot(211)
# plot(arange(P)[1::2], prob, label='coined quantum walk', color='r')
# plot(arange(P)[1::2], prob,  'o', markersize=3, color='#FFA339')
# plot(arange(P)[1::2], posnCl, label='classical random walk', color='b') # [1::2]
# xlabel('position')
# ylabel('probability')
# legend(loc='upper left', bbox_to_anchor=(1, 1))

ax2 = fig.add_subplot(111)
plot(arange(N), StandDevArr, color='r', label='coined quantum walk')
plot(arange(N), SDevCl, color='b', label='classical random walk')
xlabel('steps', fontsize=14)
ylabel('$\sigma_x$', fontsize=14)
ax2.tick_params(labelsize=14)

legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=14)

subplots_adjust(right=0.7)

show()
