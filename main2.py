import numpy as np
import sys
sys.path.append("./models/")
from controller import LinearController
from arm import GaussArm

import policy

if __name__ == '__main__':
    # initialize Arm and Rewards
    K = 8
    theta = np.array([3,1])
    arm_params = []
    for i in np.arange(1,9):
        tmp = np.array([np.cos((i * np.pi) / 4), np.sin((i * np.pi) / 4)])
        arm_params.append(tmp)
    r_sigmas = np.array([1] * K)

    arm_params = np.array(arm_params)
    g_arm = GaussArm(theta, arm_params, r_sigmas)


    epsilon = 0.3
    e_greedy = policy.LinearEpsilonGreedy(epsilon)
    e_cntl = LinearController(e_greedy, g_arm)
    e_cntl.simulate()

    # alpha, beta = 1, 1
    # thompson = policy.Thompson(alpha, beta)
    # t_cntl = Controller(thompson, g_arm)
    # t_cntl.simulate()

