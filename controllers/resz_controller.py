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
        agent_outputs = agent_outputs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, -1).mean(dim=3)
        # TODO:做一个不用均值作为 q 值的
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
