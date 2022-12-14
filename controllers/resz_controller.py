from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from controllers.basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
# 把分布式价值函数的操作单独抽离出来
class ReszMAC(BasicMAC):
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, rnd_quantiles = self.forward(ep_batch, t_ep, forward_type="approx")
        # ! 如果加入risk指数，就是在这里把得到的分布乘以风险相关的mask
        if getattr(self.args, 'static_risk', False): # 加入static_risk
            num_atoms = agent_outputs.shape[-1]
            risk_level = th.full([agent_outputs.shape[0]], 0.7)  # 0.7是临时设定的静态risk参数
            risk_atoms_num = th.ceil(risk_level * num_atoms - 1).int()
            arrange = th.arange(num_atoms).int()
            masks = (arrange <= risk_atoms_num[..., None]).float()
            masks = masks[:, None, :]
            agent_outputs = masks.cuda() * agent_outputs
        agent_outputs = agent_outputs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, -1).mean(dim=3) # 把分布变成均值，即q值，用于选取动作
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, forward_type=None):
        # agent_inputs: obs + id + last_action
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states, rnd_quantiles = self.agent(agent_inputs, self.hidden_states, forward_type=forward_type)

        return agent_outs, rnd_quantiles
    
    # def _get_input_shape(self, scheme):
    #     input_shape = scheme["obs"]["vshape"]
    #     if self.args.obs_last_action:
    #         input_shape += scheme["actions_onehot"]["vshape"][0]  # 11个动作维度
    #     if self.args.obs_agent_id:
    #         input_shape += self.n_agents

    #     return input_shape # 80 + 11 + n
