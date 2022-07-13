import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)  # schedule.eval(t) 计算时间t时的探索率
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
       
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # 根据agent_inputs(Q)和epsilon，返回动作的index
        
        self.epsilon = self.schedule.eval(t_env) # 计算探索率

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        # mask：把传入的avail_actions=0.0的动作价值设为极小
        masked_q_values = agent_inputs.clone() 
        masked_q_values[avail_actions == 0.0] = -float("inf")  # 看看不同的地图mask的是不是不一样
        # 选择动作的过程
        random_numbers = th.rand_like(agent_inputs[:, :, 0])  # 给每个ep产生一个随机数
        pick_random = (random_numbers < self.epsilon).long()  # 与epsilon比较大小 决定是否随机
        random_actions = Categorical(avail_actions.float()).sample().long()  # avail_actions只含有0/1，即在可行动作(1)中等概率选择

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]  # max返回(value, idx)
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
