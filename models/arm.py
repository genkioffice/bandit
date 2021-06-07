import numpy as np
from params import N_SIZE, N_STEP

class Arm:
    def __init__(self, means):
        self.means = means
        self.K = len(means)

    def roll(self, slot, n_item):
        return np.random.binomial(1, self.means[slot], n_item)
    
    def get_means(self):
        return self.means

    def get_K(self):
        return self.K