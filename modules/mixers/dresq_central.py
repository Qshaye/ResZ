import torch as th
import torch.nn as nn
import numpy as np


class DRESCentralMixer(nn.Module):

    def __init__(self, args):
        super(DRESCentralMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

        self.state_dim = int(np.prod(args.state_shape))
        self.input_dim = self.n_agents * self.args.n_target_quantiles + self.state_dim
        self.embed_dim = args.central_mixing_embed_dim

        # 一个三层的网络
        self.net = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, self.n_target_quantiles))
        if self.args.central_mixer == 'f':
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.embed_dim, self.n_target_quantiles))


    def forward(self, agent_qs, states, target):  # agent_qs(batch_size, episode_length, n_agents, n_quantiles)

        batch_size = agent_qs.shape[0]  # 因为传进来的是已经选过的action的 z， 所以没有n_actions维度
        episode_length = agent_qs.shape[1]

        assert states.shape == (batch_size, episode_length, self.state_dim)
        states = states.reshape(-1, self.state_dim)

        if target:
            n_rnd_quantiles = self.n_target_quantiles
        else:
            n_rnd_quantiles = self.n_quantiles

        assert agent_qs.shape == (batch_size, episode_length, self.n_agents, n_rnd_quantiles)
        agent_qs = agent_qs.reshape(-1, self.n_agents * n_rnd_quantiles)

        inputs = th.cat([states, agent_qs], dim=1)  # 这里是把state和Zi直接concat计算的，后续可以试试无单调约束的hyper_net

        Z_mixture = self.net(inputs)
        if self.args.central_mixer == 'f':
            vs = self.V(states)
            Z_mixture += vs
        Z_mixture = Z_mixture.reshape(batch_size, episode_length, 1, n_rnd_quantiles)

        return Z_mixture  # Zjt
