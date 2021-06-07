import matplotlib
import matplotlib.pyplot as plt
import seaborn
from arm import Arm
from params import N_BATCH
import numpy as np
import params
from eval import BatchEvaluator
plt.rcParams["figure.figsize"] = (8,5)
seaborn.set()

class BasicPolicy():
    def pull(self):
        pass

    
class EpsilonGreedy(BasicPolicy):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.argmax = 0
        self.e_means = 0
        self.ns_rolls = 0
        self.regrets = []


    def _set_eval(self, arm):
        self.eval = BatchEvaluator(arm)


    def pull(self, arm:Arm):
        bools = np.random.binomial(1, self.epsilon, params.N_SIZE)
        self._set_eval(arm)
        # バッチ分だけ引く
        for i in np.arange(0,params.N_SIZE, params.N_BATCH):
            locbs = bools[i:i+params.N_BATCH]
            data = arm.roll(self.argmax, params.N_BATCH)
            # pol_data = data[locbs==1]
            ass_data = data[locbs==0]
            prev_ns_rolls = self.ns_rolls
            self.calc_means(len(data[locbs==1]), arm)
            self.eval.set_evaluate(len(ass_data), self.argmax)
            diff_ns_rolls = self.ns_rolls - prev_ns_rolls
            for i_arm, times in enumerate(diff_ns_rolls):
                self.eval.set_evaluate(int(times), i_arm)
            print(self.e_means)
            print(f"iter {i}, regret: {self.eval.get_regret()}, s_mean max:{self.e_means.max()}")
            self.argmax = np.argmax(self.e_means)
            self.regrets.append(self.eval.get_regret())
        plt.plot(np.arange(params.N_SIZE/ params.N_BATCH)+1, self.regrets)
        plt.savefig(f"image/epsilon_greedy.png")

    def calc_means(self, n_data, arm:Arm):
        # initialize
        if (type(self.e_means) == int) & (type(self.ns_rolls) == int):
            self.e_means = np.zeros(arm.get_K())
            self.ns_rolls = np.zeros(arm.get_K())
        K = arm.get_K()
        if n_data == 0:
            return
        ts = n_data//K
        if ts != 0:
            for i_arm in np.arange(K):
                e_n = self.ns_rolls[i_arm]
                data = arm.roll(i_arm, ts)
                # 平均更新
                # print(self.e_means[i_arm])
                # print((-1*float(ts/(ts+e_n))) * self.e_means[i_arm])
                self.e_means[i_arm] += (-1*float(ts/(ts+e_n))) * self.e_means[i_arm]\
                                        + float(1/(ts+e_n)) * np.sum(data)
            self.ns_rolls += ts
        ts = n_data % K
        i_argmin = np.argmin(self.ns_rolls)
        for i_arm in np.arange(i_argmin, ts):
            if i_arm >= K:
                i_arm -=K
            # print(self.ns_rolls)
            # print(i_arm)
            e_n = self.ns_rolls[i_arm]
            data = arm.roll(i_arm, 1)
            self.e_means[i_arm] += (-1*float(1/(1+e_n))) * self.e_means[i_arm]\
                                        + float(1/(1+e_n)) * np.sum(data)
            self.ns_rolls[i_arm] += 1









