import copy
from components.episode_buffer import EpisodeBatch
# from modules.mixers.vdn import VDNMixer
from modules.mixers.ddn import DDNMixer
from modules.mixers.dmix import DMixer
from modules.mixers.dresq_qmix import DresQMixer
from modules.mixers.dresq_central_atten import CentralattenMixer
# from modules.mixers.qmix import QMixer
from modules.mixers.dresq_central import DRESCentralMixer
# from utils.rl_utils import build_td_lambda_targets
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop
from collections import deque
from controllers import REGISTRY as mac_REGISTRY
from utils.th_utils import get_parameters_num
from envs.one_step_matrix_game import print_matrix_status
from torch.optim import Adam


class NoCentralDResQLearner:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac   # 这个学习的一定是 Q_i
        self.logger = logger

        self.mac_params = list(mac.parameters())  # for logging
        self.params = list(self.mac.parameters())
        self.target_mac = copy.deepcopy(mac)
        self.rest_mac = copy.deepcopy(self.mac)

        self.mixer = None

        if args.mixer is not None:

            if args.mixer == "ddn":
                self.mixer = DDNMixer(args)
            elif args.mixer == "dmix":
                self.mixer = DMixer(args)
            elif args.mixer == "qmix":
                self.mixer = DresQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

            self.mixer_params = list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            self.params += list(self.mixer.parameters())

        self.rest_mixer = getattr(self.args, 'rest_mixer', None)
        self.rest_qmix = False
        if self.rest_mixer is not None:
            if args.rest_mixer == "qmix":
                self.rest_mixer = DresQMixer(args)
                self.rest_qmix = True
            elif args.rest_mixer == "atten":
                self.rest_mixer = CentralattenMixer(args) 
            elif args.rest_mixer == "dmix":
                self.rest_mixer = DMixer(args)
            elif args.rest_mixer == "ddn":
                self.rest_mixer = DDNMixer(args)
            elif args.rest_mixer == "ff":
                self.rest_mixer = DRESCentralMixer(args)  # 20220518 因为小游戏不能用atten 所以换掉
        elif args.mixer =="qmix":    # 如果没有指定rest_mix 那就是以前的版本 atten 或者双qmix
            self.rest_mixer = DresQMixer(args)
            self.rest_qmix = True
 
        self.target_rest_mixer = copy.deepcopy(self.rest_mixer)
        self.target_rest_mac = copy.deepcopy(self.rest_mac)
        self.params += list(self.target_rest_mixer.parameters())


        self.last_target_update_episode = 0

        print('Mixer Size: ')
        print(get_parameters_num(list(self.mixer.parameters()) + list(self.rest_mixer.parameters())))

        if args.optimizer == "RMSProp":
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optimizer == "Adam":
            self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            raise ValueError("Unknown Optimizer")


        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

        self.grad_norm = 1
        self.mixer_norm = 1
        self.mixer_norms = deque([1], maxlen=100)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        episode_length = rewards.shape[1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.mixer is not None:
            # Same quantile for quantile mixture
            n_quantile_groups = 1
        else:
            n_quantile_groups = self.args.n_agents

        # Calculate estimated Q-Values
        mac_out = []
        rnd_quantiles = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, agent_rnd_quantiles = self.mac.forward(batch, t=t, forward_type="policy")
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            agent_rnd_quantiles = agent_rnd_quantiles.view(batch.batch_size, n_quantile_groups, self.n_quantiles)
            mac_out.append(agent_outs)
            rnd_quantiles.append(agent_rnd_quantiles)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        rnd_quantiles = th.stack(rnd_quantiles, dim=1)  # Concat over time
        assert mac_out.shape == (batch.batch_size, episode_length + 1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        assert rnd_quantiles.shape == (batch.batch_size, episode_length + 1, n_quantile_groups, self.n_quantiles)
        rnd_quantiles = rnd_quantiles[:, :-1]
        assert rnd_quantiles.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)

        actions = actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        
        rest_mac_out = []
        self.rest_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, _ = self.rest_mac.forward(batch, t=t, forward_type="policy")
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            rest_mac_out.append(agent_outs)
        rest_mac_out = th.stack(rest_mac_out, dim=1)  # Concat over time
        assert rest_mac_out.shape == (batch.batch_size, episode_length + 1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        rest_chosen_action_qvals = th.gather(rest_mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t, forward_type="target")
            assert target_agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_agent_outs = target_agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            assert target_agent_outs.shape == (batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_mac_out.append(target_agent_outs)
        del target_agent_outs

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        assert target_mac_out.shape == (batch.batch_size, episode_length, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)

        # Mask out unavailable actions
        assert avail_actions.shape == (batch.batch_size, episode_length + 1, self.args.n_agents, self.args.n_actions)
        target_avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        target_mac_out[target_avail_actions[:, 1:] == 0] = -9999999
        
        target_rest_mac_out = []
        self.target_rest_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_rest_mac.forward(batch, t=t, forward_type="target")
            assert target_agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_agent_outs = target_agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            assert target_agent_outs.shape == (batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_rest_mac_out.append(target_agent_outs)
        del target_agent_outs

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_rest_mac_out = th.stack(target_rest_mac_out[1:], dim=1)  # Concat across time
        assert target_rest_mac_out.shape == (batch.batch_size, episode_length, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
        # Mask out unavailable actions
        target_rest_mac_out[target_avail_actions[:, 1:] == 0] = -9999999
        
        avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)  #
        

        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].mean(dim=4).max(dim=3, keepdim=True)[1]  # 用current网络选择target分布的index
            del mac_out_detach
            assert cur_max_actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
            cur_max_actions = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            rest_mac_out_detach = rest_mac_out.clone().detach()
            rest_mac_out_detach[avail_actions == 0] = -9999999
            rest_cur_max_actions = cur_max_actions
            target_rest_max_qvals = th.gather(target_rest_mac_out, 3, rest_cur_max_actions).squeeze(3)
            
        else:
            raise Exception("Use double q")

        # Mix
        Q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False)
        if self.rest_qmix:
            Q_r = self.rest_mixer(rest_chosen_action_qvals, batch["state"][:, :-1], target=False, rest=True)  #added for RESTQ
        else:
            Q_r = self.rest_mixer(chosen_action_qvals, batch["state"][:, :-1], target=False)  #added for RESTQ

        target_Q_tot = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target=True)
        target_Q_r = self.target_rest_mixer(target_rest_max_qvals, batch["state"][:, 1:], target=True)

        is_max_action = (actions == cur_max_actions).min(dim=2)[0]
        w_r = th.where(is_max_action, th.zeros_like(Q_r), th.ones_like(Q_r))  # 是最优动作时 w_r=0 不是则为1 resq(5)
        # actions = actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)
        is_max_action_target = (batch["actions"][:, 1:].unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles) == cur_max_actions).min(dim=2)[0]
        target_w_r = th.where(is_max_action_target, th.zeros_like(target_Q_r), th.ones_like(target_Q_r))  # 是最优动作时 w_r=0 不是则为1 resq(5)
        
        # Q_current = Q_tot + w_r.detach() * Q_r
        Q_current = Q_tot + w_r * Q_r
        target_max_qvals = target_Q_tot + target_w_r * target_Q_r
        
        #  targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        targets = rewards.unsqueeze(3) + (self.args.gamma * (1 - terminated)).unsqueeze(3) * target_max_qvals
        assert targets.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)


        tau = rnd_quantiles.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        targets_detach = targets.unsqueeze(3).expand(-1, -1, -1, self.n_quantiles, -1).detach()
        Q_current = Q_current.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        
        condition = targets_detach - Q_current
        quantile_weight = th.abs(tau - condition.le(0.).float())
        quantile_loss = quantile_weight * F.smooth_l1_loss(condition, th.zeros(condition.shape).cuda(), reduction='none')
        quantile_loss = quantile_loss.mean(dim=4).sum(dim=3)
        mask2 = mask.expand_as(quantile_loss)
        quantile_loss = (quantile_loss * mask2).sum() / mask2.sum()

        # 约束 Q_r>=0
        Q_r_error = th.max(Q_r, th.zeros_like(Q_r))  # 要求Q_r< 0
        mask1 = mask.unsqueeze(3).expand_as(Q_r_error)
        noopt_loss = (((Q_r_error * mask1)**2).sum()) / mask1.sum()


        loss = self.args.qmix_loss * quantile_loss + self.args.noopt_loss * noopt_loss




        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        # for Logging
        agent_norm = 0
        for p in self.mac_params:
            param_norm = p.grad.data.norm(2)
            agent_norm += param_norm.item()**2
        agent_norm = agent_norm**(1. / 2)

        mixer_norm = 0
        for p in self.mixer_params:
            param_norm = p.grad.data.norm(2)
            mixer_norm += param_norm.item()**2
        mixer_norm = mixer_norm**(1. / 2)
        self.mixer_norm = mixer_norm
        # self.mixer_norms.append(mixer_norm)

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("quantile_loss", quantile_loss.item(), t_env)
            # self.logger.log_stat("central_loss", central_loss.item(), t_env)
            if noopt_loss is not None:
                self.logger.log_stat("noopt_loss", noopt_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            self.log_stats_t = t_env

            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                # print_matrix_status(batch, self.central_mixer, mac_out)
                print_matrix_status(batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer)
                # print_matrix_status(batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer, central_mac_out=central_mac_out, rest_mac_out=rest_mac_out)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_rest_mac.load_state(self.rest_mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.target_rest_mixer.load_state_dict(self.rest_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.rest_mac.cuda()
        self.target_rest_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            self.rest_mixer.cuda()
            self.target_rest_mixer.cuda()

    # TODO: Model saving/loading is out of date!
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)

        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
