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


class LinearArm:
    def __init__(self, theta, params):
        '''
        parameters:
            theta: d-dim array.
                This value cannot be accessed by any methods but be estimated during a test.
            params: (dxk)-dim array.
                The row is corresponding to the dim of theta and k shows the vector.
        
        attributes:
            k: int.
                The number of arms.
            d: int.
                The number of params' dimensions
            ub_rewards: array-like.
                Unbiased rewards calculated by params and theta

        '''
        self.theta = theta
        self.params = params
        self.k , self.d = params.shape
        self.ub_rewards = np.dot(self.params, self.theta)

    def get_K(self):
        return self.k
        
    def roll(self, slot, n_item):
        # real_rewards = np.random.normal(loc=self.ub_rewards[slot], scale=self.sigmas[slot], size=n_item)
        # return real_rewards
        pass

class GaussArm(LinearArm):
    def __init__(self, theta, params, sigmas):
        super().__init__(theta, params)
        self.sigmas = sigmas

    def roll(self, slot, n_item):
        '''
        Rolling a selected arm and get biased rewards.
        '''
        return np.random.normal(loc=self.ub_rewards[slot], scale=self.sigmas[slot], size=n_item)

    def get_ub_rewards(self):
        return self.ub_rewards
    
    def get_sigmas(self):
        return self.sigmas