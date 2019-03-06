from numpy import *
from matplotlib.pyplot import *
from tqdm import tqdm # progress bar

N = 100 # number of random steps
P = 2*N + 1 # number of positions 
tot_trials = 100000 # for classical random walk


# classical random walk

posnCl = zeros(P)
posnCl[N] = 1
index = 0

for n in tqdm(range(tot_trials)): # n random walk
	for i in range(P): # throw coin i times
		flip = random.randint(0,2)
		if flip == 0:
			index += 1
		if flip == 1:
			index -= 1

	posnCl[index+100] += 1
	index = 0

posnCl = posnCl/(tot_trials)
posnCl = posnCl[1::2]


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


#initial state

posn0 = zeros(P)
posn0[N] = 1 # N is central position
psi0 = kron(posn0, (coin0+coin1*1j)/sqrt(2.0)) # symmetric initial coin (superposition of |0> and |1>)
psi0_0 = kron(posn0, coin0) # coin initially in |0>
psi0_1 = kron(posn0, coin1) # coin initially in |1>


# after N steps

psiN = linalg.matrix_power(U, N).dot(psi0)
psiN_0 = linalg.matrix_power(U, N).dot(psi0_0)
psiN_1 = linalg.matrix_power(U, N).dot(psi0_1)


# measurement

prob = empty(P)
for k in range(P):
	posn = zeros(P)
	posn[k] = 1
	M_hat_k = kron(outer(posn, posn), eye(2))
	proj = M_hat_k.dot(psiN)
	prob[k] = proj.dot(proj.conjugate()).real
prob = prob[::2]

prob_0 = empty(P)
for i in range(P):
	posn = zeros(P)
	posn[i] = 1
	M_hat_i = kron(outer(posn, posn), eye(2))
	proj = M_hat_i.dot(psiN_0)
	prob_0[i] = proj.dot(proj.conjugate()).real
prob_0 = prob_0[::2]

prob_1 = empty(P)
for l in range(P):
	posn = zeros(P)
	posn[l] = 1
	M_hat_l = kron(outer(posn, posn), eye(2))
	proj = M_hat_l.dot(psiN_1)
	prob_1[l] = proj.dot(proj.conjugate()).real
prob_1 = prob_1[::2]



# plot

x_ax = arange(P)-100
x_ax = x_ax[::2]

fig = figure(figsize=(14,7), dpi=200)




ax = fig.add_subplot(311)

# title('Random quantum and classical walks')

plot(x_ax, prob, color='r')
plot(x_ax, prob, 'o', markersize=3, color=(1,0.65,0))
plot(x_ax[1::], posnCl, color='b')
# plot(arange(P)-100, posnCl, 'o', color=(1,0.6,0,1), markersize=3)
setp(ax.get_xticklabels(), visible=False)
setp(gca(), yticks=(0.01, 0.04, 0.07, 0.10), xticks=(-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100))
ax.tick_params(labelsize=14)




ax_0 = fig.add_subplot(312)

plot(x_ax, prob_0, color='r', label='quantum walk')
plot(x_ax, prob_0, 'o', markersize=3, color=(1,0.65,0))
plot(x_ax[1::], posnCl, color='b', label='calssical walk')
# plot(arange(P)-100, posnCl, 'o', color=(1,0.6,0,1), markersize=3)
setp(ax_0.get_xticklabels(), visible=False)
setp(gca(), yticks=(0.02, 0.06, 0.10, 0.14), xticks=(-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100))
legend(loc=(1.02,1.7), fontsize=14)
ax_0.tick_params(labelsize=14)

ylabel('Probability', fontsize=14)




ax_1 = fig.add_subplot(313)

plot(x_ax, prob_1, color='r')
plot(x_ax, prob_1, 'o', markersize=3, color=(1,0.65,0))
plot(x_ax[1::], posnCl, color='b')
# plot(arange(P)-100, posnCl, 'o', color=(1,0.6,0,1), markersize=3)
setp(gca(), yticks=(0.02, 0.06, 0.10, 0.14), xticks=(-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100))
ax_1.tick_params(labelsize=14)

xlabel('Postion', fontsize=14)


subplots_adjust(hspace=0, right=0.8)

show()