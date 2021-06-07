import numpy as np
N_SIZE = 10000
N_STEP = 100

X = np.array([])
X1, X2, X3, X4 = [0.1], [0.05], [0.02], [0.01]
X = X1 + X2*3 + X3*3 + X4*3

EPSILON = 0.2
K = len(X)
regs = []
maxv = max(X)
for step in np.arange(N_STEP):
    means = []
    reg = 0
    rep = int(N_SIZE/K * EPSILON)
    for slot in np.arange(K):
        v = np.random.binomial(1, X[slot], rep)
        reg += rep * (maxv - X[slot])
        means.append(v.mean())
    hatv = max(means)
    print(means)
    for 

