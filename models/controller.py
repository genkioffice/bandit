import numpy as np
from policy import BasicPolicy
from arm import Arm

class Controller:
    def __init__(self, policy:BasicPolicy, means:np.array):
        self.policy = policy
        self.means = means
        self._set()
        
    def _set(self):
        self.arm = Arm(self.means)

    def simulate(self):
        self.policy.pull(self.arm)

class LinearController:
    def __init__(self, policy:BasicPolicy, arm:Arm):
        self.policy = policy
        self.arm = arm

    def simulate(self, plot=True):
        return self.policy.pull(self.arm, plot=plot)
