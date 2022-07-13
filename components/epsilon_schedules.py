import numpy as np


class DecayThenFlatSchedule():

    # 递减策略，用于action_selector中的class EpsilonGreedyActionSelector()

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        # epsilon_start: 1.0
        # epsilon_finish: 0.05    
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length # 计算线性递减时每个时间片递减的探索率
        self.decay = decay 

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1
            # 计算 e^(-(time_length/exp_scaling))=finish 也就是时间最长时刚好衰减到最小值
            
    # 根据当前时间，来计算探索率epsilon
    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass
