import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """ 
    RNNAgent:智能体的网络
    in：input_shape(obs_shape + id_oneHot + lastAction_oneHot)
    out：n_actions 
    """

    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        # rnn_hidden_dim: 64 # Size of hidden state for default rnn agent
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # tensor.new(shape)会创建一个dtype和shape与tensor一致,形状为shape的新tensor
        # .zero_() 元素全设为0
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()  # [1, 64]

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
