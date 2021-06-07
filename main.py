import numpy as np
import sys
sys.path.append("models/")
from controller import Controller

import policy

if __name__ == '__main__':
    X = np.array([])
    X1, X2, X3, X4 = [0.1], [0.05], [0.02], [0.01]
    X = X1 + X2*3 + X3*3 + X4*3
    print(X)
    epsilon = 0.3
    e_greedy = policy.EpsilonGreedy(epsilon)
    e_cntl = Controller(e_greedy, X)
    e_cntl.simulate()

    alpha, beta = 1, 1
    thompson = policy.Thompson(alpha, beta)
    t_cntl = Controller(thompson, X)
    t_cntl.simulate()

