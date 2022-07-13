from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

# 这里同时适配Q和Z
# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args

        # 调用后根据观测信息obs_shape 得到input_shape, 可选是否加入id和last action
        input_shape = self._get_input_shape(scheme)
        # 该函数会调用modules -> angents中的代码 创建agent
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type  # "q"

        # qmix中 action_selector: "epsilon_greedy"
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # 用时间t 选择当前要更新的ep
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if self.args.agent == "iqn_rnn":
            agent_outputs, rnd_quantiles = self.forward(ep_batch, t_ep, forward_type="approx")
        else:
            agent_outputs = self.forward(ep_batch, t_ep, forward_type=test_mode)

        if self.args.agent == "iqn_rnn":  # 把 Z 变成 Q -> 直接mean
            agent_outputs = agent_outputs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, -1).mean(dim=3)
            # TODO:做一个不用均值作为 q 值的
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, forward_type=None):
        # agent_inputs: obs + id + last_action
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if self.args.agent == "iqn_rnn":  # agent_outs是个分布，同时还要输出quantiles
            agent_outs, self.hidden_states, rnd_quantiles = self.agent(agent_inputs, self.hidden_states, forward_type=forward_type)
        else:
            # self.agent是RNNAgent的实例化 调用Agent的forward输出q,h
            # h更新self.hidden_states，以便下次传入agent网络
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # COMA algs Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            # getattr(a, 'weight')       返回a的weight值 类似a['weight']
            # getattr(a, 'weight', 500)  返回a的weight值 若没有该属性，返回500(缺省值)
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not forward_type:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0
        if self.args.agent == "iqn_rnn":
            return agent_outs, rnd_quantiles
        else:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)  # [batch_size, n_agent, n_action]

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        # [1, 64] -> [1, 1, 64] -> [1, 5, 64]
        # tensor.expand() 可以将维度为1的dim 复制为多维度

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())  # 加载权重

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # 调用Episode中__getitem__方法  加入观测obs
        # 加入可观测的 action
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])    # t-1 的action
        # 加入智能体id信息
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))   # bs: batch_size

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]  # 11个动作维度
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape # 80 + 11 + n
