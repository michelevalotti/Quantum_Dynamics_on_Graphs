import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # progress bar

N = 20 # size of lattice
initialPosn = 200

Gcoin = 0.5 * np.array([[-1,1,1,1],[1,-1,1,1],[1,1,-1,1],[1,1,1,-1]])

psi0 = np.zeros(N**2)
psi0[initialPosn] = 1
psi0 = np.kron(psi0,np.ones((4)))

