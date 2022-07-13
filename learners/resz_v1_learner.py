#! trying to find a simple version without hurting the performance
import copy
from components.episode_buffer import EpisodeBatch
# from modules.mixers.vdn import VDNMixer
from modules.mixers.ddn import DDNMixer
from modules.mixers.dmix import DMixer
from modules.mixers.datten import DattenMixer
from modules.mixers.dresq_qmix import DresQMixer
# from modules.mixers.qmix import QMixer
from modules.mixers.dresq_central_atten import CentralattenMixer
from modules.mixers.dresq_central import DRESCentralMixer
from utils.rl_utils import build_distributional_td_lambda_targets
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop
# from collections import deque
from controllers import REGISTRY as mac_REGISTRY
from utils.th_utils import get_parameters_num
from envs.one_step_matrix_game import print_matrix_status
from torch.optim import Adam

# 0619 reduce Zr_i

class ResZ_todo:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())  # for logging
        self.params = list(self.mac.parameters())

        # self.rest_mac = copy.deepcopy(self.mac)  # added for Q_r_i in RESQ
        # self.params += list(self.rest_mac.parameters())  # added for RESTQ

        self.mixer = None

        if args.mixer is not None:

            if args.mixer == "ddn":
                self.mixer = DDNMixer(args)
            elif args.mixer == "qmix":
                self.mixer = DresQMixer(args)
            elif args.mixer == "dmix":
                self.mixer = DMixer(args)
            if args.mixer == "datten":
                self.mixer = DattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

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
                self.rest_mixer = DRESCentralMixer(args)
        elif args.mixer == "qmix":  # 如果没有指定rest_mix 那就是以前的版本 atten 或者双qmix
            self.rest_mixer = DresQMixer(args)
            self.rest_qmix = True
        else:
            self.rest_mixer = CentralattenMixer(args)
        self.params += list(self.rest_mixer.parameters())

        # Central Q for Qjt and Qjt_hat
        # 用central_agent学真实的Qjt, 再与 Qtot+Qr 作loss
        self.central_mac = None
        assert args.central_mac == "base_central_mac"
        self.central_mac = mac_REGISTRY[args.central_mac](scheme, args)
        self.target_central_mac = copy.deepcopy(self.central_mac)
        self.params += list(self.central_mac.parameters())

        self.central_mixer = CentralattenMixer(args)

        self.params += list(self.central_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)

        self.last_target_update_episode = 0

        print('Mixer Size: ')
        print(get_parameters_num(list(self.mixer.parameters()) + list(self.central_mixer.parameters()) + list(self.rest_mixer.parameters())))

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

        # rest Zi
        '''
        rest_mac_out = []
        self.rest_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, _ = self.rest_mac.forward(batch, t=t, forward_type="policy")
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            rest_mac_out.append(agent_outs)
        rest_mac_out = th.stack(rest_mac_out, dim=1)  # Concat over time
        assert rest_mac_out.shape == (batch.batch_size, episode_length + 1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        rest_chosen_action_qvals = th.gather(rest_mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        '''

        # Central MAC stuff
        central_mac_out = []
        central_rnd_quantiles = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, agent_rnd_quantiles = self.central_mac.forward(batch, t=t, forward_type="policy")
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            agent_rnd_quantiles = agent_rnd_quantiles.view(batch.batch_size, n_quantile_groups, self.n_quantiles)
            central_mac_out.append(agent_outs)
            central_rnd_quantiles.append(agent_rnd_quantiles)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        central_rnd_quantiles = th.stack(central_rnd_quantiles, dim=1)
        central_chosen_action_qvals = th.gather(central_mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        target_central_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, _ = self.target_central_mac.forward(batch, t=t, forward_type="target")
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            target_central_mac_out.append(agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_central_mac_out = th.stack(target_central_mac_out[:], dim=1)

        # Mask out unavailable actions
        target_avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        target_central_mac_out[target_avail_actions[:, :] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:  # Get actions that maximise live Q (for double q-learning)
            avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)
            mac_out_detach = mac_out.clone().detach()  # mac_out batch_size, seq_length, n_agents, n_commands
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, :].mean(dim=4).max(dim=3, keepdim=True)[1]
        else:
            cur_max_actions = target_central_mac_out.mean(dim=4).max(dim=3, keepdim=True)[1]  # calculate Q -> argmax Q

        cur_max_actions = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        target_max_qvals = th.gather(target_central_mac_out, 3, cur_max_actions).squeeze(3)

        # Mix
        Q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False)
        if self.rest_qmix:
            Q_r = self.rest_mixer(chosen_action_qvals, batch["state"][:, :-1], target=False, rest=True)  #added for RESTQ
        else:
            Q_r = self.rest_mixer(chosen_action_qvals, batch["state"][:, :-1], target=False)

        Q_r = -1 * Q_r.abs()

        central_chosen_qvals = self.central_mixer(central_chosen_action_qvals, batch["state"][:, :-1], target=False)  #
        target_max_qvals = self.target_central_mixer(target_max_qvals, batch["state"], target=True)

        #  targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        if self.args.td_lambda == 0:
            targets = rewards.unsqueeze(3) + (self.args.gamma * (1 - terminated)).unsqueeze(3) * target_max_qvals[:, 1:]
        else:
            targets = build_distributional_td_lambda_targets(rewards, terminated, mask, target_max_qvals, self.args.n_agents, self.args.gamma, self.args.td_lambda)
        assert targets.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)


        tau = rnd_quantiles.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        targets_detach = targets.unsqueeze(3).expand(-1, -1, -1, self.n_quantiles, -1).detach()
        # L_r loss
        is_max_action = (actions == cur_max_actions[:, 1:]).min(dim=2)[0]  # 判断多个 agent 选择的 action 组合是否 argmax Qtot的，min:只要有一个是False就是False
        w_r = th.where(is_max_action, th.zeros_like(Q_r), th.ones_like(Q_r))  # 是最优动作时 w_r=0 不是则为1 resq(5)
        w_to_use = w_r.mean().item()  # Average of wr for logging
        Q_jt = Q_tot + w_r.detach() * Q_r
        Q_jt = Q_jt.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        condition = targets_detach - Q_jt
        quantile_weight = th.abs(tau - condition.le(0.).float())
        quantile_loss = quantile_weight * F.smooth_l1_loss(condition, th.zeros(condition.shape).cuda(), reduction='none')
        quantile_loss = quantile_loss.mean(dim=4).sum(dim=3)
        mask2 = mask.expand_as(quantile_loss)
        quantile_loss = (quantile_loss * mask2).sum() / mask2.sum()

        # central Quantile loss
        targets = targets.unsqueeze(3).expand(-1, -1, -1, self.n_quantiles, -1)
        central_chosen_qvals = central_chosen_qvals.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        condition = targets_detach - central_chosen_qvals
        # 1-tau/tau
        quantile_weight = th.abs(tau - condition.le(0.).float())
        central_loss = quantile_weight * F.smooth_l1_loss(condition, th.zeros(condition.shape).cuda(), reduction='none')
        central_loss = central_loss.mean(dim=4).sum(dim=3)
        mask_central = mask.expand_as(central_loss)
        central_loss = (central_loss * mask_central).sum() / mask_central.sum()

        loss = 1 * quantile_loss + 1 * central_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        # for Logging
        agent_norm = 0
        for p in self.mac_params:
            param_norm = p.grad.data.norm(2)
            agent_norm += param_norm.item()**2
        agent_norm = agent_norm**(1. / 2)

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("quantile_loss", quantile_loss.item(), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            self.logger.log_stat("w_to_use", w_to_use, t_env)
            self.log_stats_t = t_env

            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                # print_matrix_status(batch, self.central_mixer, mac_out)
                print_matrix_status(batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer)
                # print_matrix_status(batch, self.mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=True, wqmix_central_mixer=self.central_mixer, rest_mixer=self.rest_mixer, central_mac_out=central_mac_out, rest_mac_out=rest_mac_out)

    def _update_targets(self):
        if self.central_mac is not None:
            self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        # self.rest_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.rest_mixer.cuda()
        if self.central_mac is not None:
            self.central_mac.cuda()
            self.target_central_mac.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

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
