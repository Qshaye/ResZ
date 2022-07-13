import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DresQMixer(nn.Module):

    def __init__(self, args):
        super(DresQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim  # mixing_embed_dim: 32
        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

        # 根据state计算mix的权重
        # hypernet_layers: 2
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
            
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, target, rest=False):
        bs = agent_qs.size(0)
        if target:
            n_rnd_quantiles = self.n_target_quantiles
        else:
            n_rnd_quantiles = self.n_quantiles
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents, n_rnd_quantiles)
        agent_qs = agent_qs.permute(0, 3, 1, 2)
        # First layer
        w1 = self.hyper_w_1(states)
        if not rest:
            w1 = th.abs(w1)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w1 = w1.unsqueeze(1).expand(-1, n_rnd_quantiles, -1, -1)
        b1 = b1.reshape(-1, 1, self.embed_dim).unsqueeze(1).expand(-1, n_rnd_quantiles, -1, -1)
        hidden = F.elu(th.matmul(agent_qs, w1) + b1)
        # Second layer
        w_final = self.hyper_w_final(states)
        if not rest:
            w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.embed_dim, 1).unsqueeze(1).expand(-1, n_rnd_quantiles, -1, -1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1).unsqueeze(1).expand(-1, n_rnd_quantiles, -1, -1)
        # Compute final output
        y = th.matmul(hidden, w_final) + v
        # Reshape and return
        Z_tot = y.reshape(bs, -1, 1, n_rnd_quantiles)
        return Z_tot
