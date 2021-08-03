import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
import seaborn
from arm import Arm
import copy
from params import N_BATCH
import numpy as np
import params
from eval import BatchEvaluator, NaiveEvaluator, BatchLinearEvaluator
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
        self.ns_rolls_exp = 0
        self.ns_rolls = []
        self.regrets = []


    def _set_eval(self, arm):
        self.eval = BatchEvaluator(arm)
        self.ns_rolls = np.zeros(arm.get_K())


    def pull(self, arm:Arm):
        # 1の出る確率がepsilon
        bools = np.random.binomial(1, self.epsilon, params.N_SIZE)
        self._set_eval(arm)
        # バッチ分更新する
        for i in np.arange(0,params.N_SIZE, params.N_BATCH):
            locbs = bools[i:i+params.N_BATCH]
            data = arm.roll(self.argmax, params.N_BATCH)
            # locbsが0のとき、argmaxの物を引く。1のとき、探索が実行される。
            ass_data = data[locbs==0]
            prev_ns_rolls = copy.deepcopy(self.ns_rolls_exp)
            # 探索part: パラメータの更新
            self.calc_means(len(data[locbs==1]), arm)
            self.eval.set_evaluate(len(ass_data), self.argmax)
            # 活用part: パラメータを更新せず、最適な腕を引く
            self.ns_rolls[self.argmax] += len(ass_data)
            diff_ns_rolls = self.ns_rolls_exp - prev_ns_rolls
            # regret更新
            for i_arm, times in enumerate(diff_ns_rolls):
                self.eval.set_evaluate(int(times), i_arm)
                self.ns_rolls[i_arm] += int(times)
            print(f"iter {i}, regret: {self.eval.get_regret()}, s_mean max:{self.e_means.max()}")
            self.argmax = np.argmax(self.e_means)
            self.regrets.append(self.eval.get_regret())
            # print(self.ns_rolls_exp)
        plt.plot(np.arange(params.N_SIZE/ params.N_BATCH)+1, self.regrets)
        plt.savefig(f"image/epsilon_greedy.png")
        print(self.ns_rolls)

    # 標本平均の計算
    def calc_means(self, n_data, arm:Arm):
        # initialize
        if (type(self.e_means) == int) & (type(self.ns_rolls_exp) == int):
            self.e_means = np.zeros(arm.get_K())
            self.ns_rolls_exp = np.zeros(arm.get_K())
        K = arm.get_K()
        if n_data == 0:
            return
        ts = n_data//K
        if ts != 0:
            # 各armでts回引く
            for i_arm in np.arange(K):
                e_n = self.ns_rolls_exp[i_arm]
                data = arm.roll(i_arm, ts)
                # 平均更新
                self.e_means[i_arm] += (-1*float(ts/(ts+e_n))) * self.e_means[i_arm]\
                                        + float(1/(ts+e_n)) * np.sum(data)
            self.ns_rolls_exp += ts
        # K回以下の残りについて標本平均を更新する
        ts = n_data % K
        i_argmin = np.argmin(self.ns_rolls_exp)
        for i_arm in np.arange(i_argmin, ts):
            if i_arm >= K:
                i_arm %=K
            e_n = self.ns_rolls_exp[i_arm]
            data = arm.roll(i_arm, 1)
            self.e_means[i_arm] += (-1*float(1/(1+e_n))) * self.e_means[i_arm]\
                                        + float(1/(1+e_n)) * np.sum(data)
            self.ns_rolls_exp[i_arm] += 1


class LinearEpsilonGreedy(BasicPolicy):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.e_theta = 0
        # thetaの推定のための推定量
        self.A, self.b = 0, 0
        self.ns_rolls = []
        self.last_pulled = 0
        self.argmax = 0
        self.ns_rolls_exp = 0
        self.regrets = []


    def _set_eval(self, arm):
        self.eval = BatchLinearEvaluator(arm)
        self.ns_rolls = np.zeros(arm.get_K())


    def pull(self, arm:Arm):
        # 1の出る確率がepsilon
        bools = np.random.binomial(1, self.epsilon, params.N_SIZE)
        self._set_eval(arm)
        # バッチ分だけ更新する
        for i in np.arange(0,params.N_SIZE, params.N_BATCH):
            locbs = bools[i:i+params.N_BATCH]
            e_regrets = arm.roll(self.argmax, params.N_BATCH)
            # locbsが0のとき、argmaxの物を引く。1のとき、探索が実行される。
            # 準備
            prev_ns_rolls = copy.deepcopy(self.ns_rolls_exp)
            # 探索part
            self.estimate(len(locbs[locbs==1]), arm)
            # 活用part: ns_rollsのargmax部を伸ばしておくだけで、下のfor文で値を更新する
            ass_data = e_regrets[locbs==0]
            self.ns_rolls[self.argmax] += len(ass_data)
            # 活用partではパラメタ更新しない
            diff_ns_rolls = self.ns_rolls_exp - prev_ns_rolls
            # regret更新
            for i_arm, times in enumerate(diff_ns_rolls):
                self.eval.set_evaluate(int(times), i_arm)
                self.ns_rolls[i_arm] += int(times)
            print(f"iter {i}, regret: {self.eval.get_regret()}, theta_hat:{self.theta_hat}")
            self.argmax = np.argmax(self.e_rewards)
            self.regrets.append(self.eval.get_regret())

        plt.plot(np.arange(params.N_SIZE/ params.N_BATCH)+1, self.regrets)
        plt.savefig(f"image/linear_epsilon_greedy.png")
        print(self.ns_rolls)

    # thetaを推定し、regret更新のための情報を保存する
    def estimate(self, n_data, arm:Arm):
        # initialize
        if (type(self.e_theta) == int) & (type(self.ns_rolls_exp) == int):
            self.e_theta = np.zeros(arm.get_K())
            # このベクトルは各アームが何回実行されたかを保存する
            self.ns_rolls_exp = np.zeros(arm.get_K())
        K = arm.get_K()
        if n_data == 0:
            return
        ts = n_data//K
        if ts != 0:
            # 各armでts回引く
            for i_arm in np.arange(K):
                self.A += np.dot(arm.params[i_arm].reshape((len(arm.params[i_arm]), 1)), arm.params[i_arm].reshape(1, len(arm.params[i_arm]))) * ts
                self.b += np.sum(arm.roll(i_arm, ts)) * arm.params[i_arm]            
            self.ns_rolls_exp += ts
        # K回以下の残りについて標本平均を更新する
        ts = n_data % K
        for i_arm in np.arange(self.last_pulled, ts + self.last_pulled):
            if i_arm >= K:
                i_arm %= K
            rw = arm.roll(i_arm, 1)
            self.A += np.dot(arm.params[i_arm].reshape((len(arm.params[i_arm]), 1)), arm.params[i_arm].reshape(1, len(arm.params[i_arm])))
            self.b += rw * arm.params[i_arm]
            self.ns_rolls_exp[i_arm] += 1
            self.last_pulled = i_arm + 1

        self.theta_hat = np.dot(np.linalg.inv(self.A), self.b.reshape((len(self.b), 1)))
        self.e_rewards = [np.dot(self.theta_hat.T, v.reshape((len(v),1))) for v in arm.params]
        self.argmax = np.argmax(self.e_rewards)


class Thompson(BasicPolicy):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.regrets = []
        self.ns_rolls = []
        pass

    def _set(self, arm:Arm):
        self.ns = np.zeros(arm.get_K())
        self.ms = np.zeros(arm.get_K())
        self.ns_rolls = np.zeros(arm.get_K())
        self.eval = BatchEvaluator(arm)

    def pull(self, arm:Arm):
        self._set(arm)
        for i in np.arange(0,params.N_SIZE, params.N_BATCH):
            e_means = np.zeros((params.N_BATCH,arm.get_K()))
            for t, i_arm in enumerate(np.arange(arm.get_K())):
                n, m = self.ns[i_arm], self.ms[i_arm]
                e_means[:,i_arm] = np.random.beta(self.alpha + m, self.beta + n - m, params.N_BATCH)
            e_argmaxs = np.array(list(map(lambda x: np.argmax(x), e_means)))
            counts = Counter(e_argmaxs)
            for i_arm, count in counts.items():
                x = arm.roll(i_arm, count)
                # 事後分布更新
                self.ns[i_arm] += count
                self.ms[i_arm] += x.sum()
                self.eval.set_evaluate(count, i_arm)
                self.ns_rolls[i_arm] += count
            self.regrets.append(self.eval.get_regret())
            # print(e_argmax, self.eval.get_regret())
            
        plt.figure()
        plt.plot(np.arange(params.N_SIZE/ params.N_BATCH)+1, self.regrets)
        plt.savefig(f"image/thompson.png")
        print(self.ns_rolls)

        
class LinearThompson(BasicPolicy):
    def __init__(self, tau, xi):
        self.tau = tau
        self.xi = xi
        self.regrets = []
        self.ns_rolls = []
        pass

    def _set(self, arm:Arm):
        self.ns = np.zeros(arm.get_K())
        self.s_means = np.zeros(arm.get_K())
        self.ns_rolls = np.zeros(arm.get_K())
        self.sigmas = arm.get_sigmas()
        self.eval = BatchLinearEvaluator(arm)

    def pull(self, arm:Arm):
        self._set(arm)
        for i in np.arange(0,params.N_SIZE, params.N_BATCH):
            x_means = np.zeros((params.N_BATCH,arm.get_K()))
            # 事後分布からサンプリング(初回は事前分布からサンプリングする)
            for t, i_arm in enumerate(np.arange(arm.get_K())):
                n = self.ns[i_arm]
                sigma = self.sigmas[i_arm]
                s_mean = self.s_means[i_arm]
                x_mean_hat = (float(n/sigma)/(float(n/sigma) + (1/self.tau)))* s_mean + (float(1/self.tau)/(float(n/sigma) + (1/self.tau))) * self.xi
                x_sigma = np.sqrt(1/(float(n/sigma) + (1/self.tau)))
                x_means[:,i_arm] = np.random.normal(loc=x_mean_hat, scale=x_sigma, size=params.N_BATCH)
            e_argmaxs = np.array(list(map(lambda x: np.argmax(x), x_means)))
            counts = Counter(e_argmaxs)
            for i_arm, count in counts.items():
                x = arm.roll(i_arm, count)
                # 事後分布更新(事後分布で与えられるパラメタの更新)
                prev_n = self.ns[i_arm]
                self.ns[i_arm] += count
                self.s_means[i_arm] = float(prev_n/(self.ns[i_arm]))  * self.s_means[i_arm] + float(1/self.ns[i_arm]) * x.sum()
                self.eval.set_evaluate(count, i_arm)
                self.ns_rolls[i_arm] += count
            self.regrets.append(self.eval.get_regret())
            print(f"iter {i}, regret: {self.eval.get_regret()}, means:{self.s_means}")
            
        plt.figure()
        plt.plot(np.arange(params.N_SIZE/ params.N_BATCH)+1, self.regrets)
        plt.savefig(f"image/linear_thompson.png")
        np.set_printoptions(suppress=True)
        print(self.ns_rolls)



class LinearUCB(BasicPolicy):
    def __init__(self):
        self.theta = 0
        self.A = np.array([[1e-5,0],[0,1e-5]])
        self.b = 0
        self.alpha = np.sqrt(2 * np.log(20)) # 有意水準0.05
        self.regrets = []


    def _set(self, arm:Arm):
        self.sigmas = arm.get_sigmas()
        self.K = arm.get_K()
        self.ns_rolls = np.zeros(self.K)
        self.ucb = np.zeros(self.K)
        self.eval = BatchLinearEvaluator(arm)
        
    
    def pull(self, arm:Arm):
        self._set(arm)
        for i_arm in np.arange(arm.get_K()):
            reward = arm.roll(i_arm, 1)
            self.estimate(reward, i_arm, arm)
            self.ucb[i_arm] = self.calc_ucb(i_arm, arm)
            self.ns_rolls[i_arm] = 1
            self.eval.set_evaluate(1, i_arm)
        i_argmax = np.argmax(self.ucb)
        for t in np.arange(params.N_BATCH):
            # step初回は1回分引いたことにする
            if t == 0:
                rewards = arm.roll(i_argmax, params.N_STEP - 1)
                self.eval.set_evaluate(params.N_STEP - 1, i_argmax)
            else:
                rewards = arm.roll(i_argmax, params.N_STEP)
                self.eval.set_evaluate(params.N_STEP, i_argmax)
            self.estimate(rewards, i_argmax, arm)
            self.ns_rolls[i_argmax] += params.N_STEP
            for i_arm in np.arange(self.K):
                self.ucb[i_arm] = self.calc_ucb(i_arm, arm)
            i_argmax = np.argmax(self.ucb)
            self.regrets.append(self.eval.get_regret())
            # print(f"ucb: {self.ucb}")

        plt.plot(np.arange(params.N_STEP)+1, self.regrets)
        plt.savefig(f"image/linear_ucb_greedy.png")
        np.set_printoptions(suppress=True)
        print(self.ns_rolls)


    # thetaの推定(future: 逆行列の計算の効率化)
    def estimate(self, rewards, i_arm, arm:Arm):
        n = len(rewards)
        self.A += np.dot(arm.params[i_arm].reshape((len(arm.params[i_arm]), 1)), arm.params[i_arm].reshape(1, len(arm.params[i_arm]))) * n
        self.b += np.sum(rewards) * arm.params[i_arm]
        self.A_inv = np.linalg.inv(self.A)
        self.theta = np.dot(self.A_inv, self.b.reshape((len(self.b), 1)))

    # ucbの計算
    def calc_ucb(self, i_arm, arm):
        d = len(self.theta)
        # print(np.dot(arm.params[i_arm].resize((1,d)),self.A_inv))
        a = arm.params[i_arm].reshape((1,d))
        a_inv = arm.params[i_arm].reshape((d,1))
        tmp = np.dot(a, self.A_inv)
        d = np.dot(tmp,a_inv)

        return np.dot(a, self.theta) + self.sigmas[i_arm] * self.alpha * np.sqrt(d)
