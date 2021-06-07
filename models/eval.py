from arm import Arm

class BasicEvaluator:
    def __init__(self, arm:Arm):
        self.regret = 0.0
        self.arm = arm
        self._set()
    
    def _set(self):
        self.means = self.arm.get_means()
        self.mean_max = max(self.means)
        self.K = len(self.means)

    def evaluate(self):
        pass

    def get_regret(self):
        return self.regret


class BatchEvaluator(BasicEvaluator):
    def __init__(self, arm:Arm):
        self.n = 0
        return super().__init__(arm)    


    def set_evaluate(self, n_data, e_argmax):
        self.regret += n_data * (self.mean_max - self.means[e_argmax])
        
        
