import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop
from collections import deque
from controllers import REGISTRY as mac_REGISTRY
from utils.th_utils import get_parameters_num
from envs.one_step_matrix_game import print_matrix_status
from torch.optim import Adam


class RestQLearner:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())  # for logging
        self.params = list(self.mac.parameters())
        # ! 可以去掉试试
        self.rest_mac = copy.deepcopy(self.mac)  # added for Q_r_i in RESTQ

        self.params += list(self.rest_mac.parameters())  # added for RESTQ
        self.rest_mixer = QMixerCentralFF(args)  # TODO 无单调约束的分布mixer 学习Q_r

        self.mixer = None
        assert args.mixer is not None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

            self.mixer_params = list(self.mixer.parameters())
            self.params += list(self.mixer.parameters())

        # Central mac for Qjt

        self.central_mac = None
        assert args.central_mac == "basic_central_mac"
        self.central_mac = mac_REGISTRY[args.central_mac](scheme, args)
        self.target_central_mac = copy.deepcopy(self.central_mac)
        self.params += list(self.central_mac.parameters())

        if self.args.central_mixer in ["ff", "atten", "vdn"]:   # TODO 无单调约束的分布mixer
            if self.args.central_loss == 0:
                self.central_mixer = self.mixer
                self.central_mac = self.mac
                self.target_central_mac = self.target_mac
            else:
                if self.args.central_mixer == "ff":
                    self.central_mixer = QMixerCentralFF(args)  # TODO 无单调约束的分布mixer
                elif self.args.central_mixer == "vdn":
                    self.central_mixer = VDNMixer()
                else:
                    raise Exception("Error with central_mixer")
        else:
            raise Exception("Error with qCentral")

        self.params += list(self.central_mixer.parameters())
        self.params += list(self.rest_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)

        self.last_target_update_episode = 0

        print('Mixer Size: ')
        print(get_parameters_num(list(self.mixer.parameters()) + list(self.central_mixer.parameters()) + list(self.rest_mixer.parameters())))

        if hasattr(self, "optimizer"):
            if getattr(self, "optimizer") == "Adam":
                self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.grad_norm = 1
        self.mixer_norm = 1
        self.mixer_norms = deque([1], maxlen=100)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals = chosen_action_qvals_agents

        # ! 可以去掉试试
        rest_mac_out = []
        self.rest_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            rest_agent_outs = self.rest_mac.forward(batch, t=t)
            rest_mac_out.append(rest_agent_outs)
        rest_mac_out = th.stack(rest_mac_out, dim=1)  # Concat over time
        rest_chosen_action_qvals = th.gather(rest_mac_out[:, :-1], dim=3, index=actions).repeat(1, 1, 1, self.args.central_action_embed).squeeze(3)

        # Max over target Q-Values
        if self.args.double_q:  # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()  # mac_out batch_size, seq_length, n_agents, n_commands
            mac_out_detach[avail_actions == 0] = -9999999
            _, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)  # (max, max_indices) = torch.max(input, dim, keepdim=False)
        else:
            raise Exception("Use double q")

        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time

        central_chosen_action_qvals = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)  # Remove the last dim

        central_target_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time

        central_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl

        # Use the Qmix max actions
        # 后面的build_td_lambda_targets会把多的一个time_step去掉
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:, :], 3, cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        Q_r = self.rest_mixer(rest_chosen_action_qvals, batch["state"][:, :-1])  #added for RESTQ
        Q_joint = self.central_mixer(central_chosen_action_qvals, batch["state"][:, :-1])  #这个就是Q^*(s,\tau, u)

        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals, batch["state"])
        # We use the calculation function of sarsa lambda to approximate q star lambda
        Q_joint_hat = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, self.args.n_agents, self.args.gamma, self.args.td_lambda)  # Qjt_hat

        # central Q loss
        central_td_error = (Q_joint - Q_joint_hat.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error**2).sum() / mask.sum()

        # QMIX loss with weighting
        # 判断多个 agent 选择的 action 组合是否 argmax Qtot的，min:只要有一个是False就是False
        is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
        w_r = th.where(is_max_action, th.zeros_like(central_td_error), th.ones_like(central_td_error))  # 是最优动作时 w_r=0 不是则为1 resq(5)
        w_to_use = w_r.mean().item()  # Average of ws for logging
        td_error = (chosen_action_qvals + w_r.detach() * Q_r - (Q_joint_hat.detach()))  # Qjt = Q_tot + w_r * Q_r
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask  # 0-out the targets that came from padded data
        qmix_loss = (masked_td_error**2).sum() / mask.sum()

        # 约束 Q_r>=0
        Q_r_error = th.max(Q_r, th.zeros_like(Q_r))  # 要求Q_r< 0
        noopt_loss = (((Q_r_error * mask)**2).sum()) / mask.sum()

        loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss + self.args.noopt_loss * noopt_loss

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
            self.logger.log_stat("qmix_loss", qmix_loss.item(), t_env)
            if noopt_loss is not None:
                self.logger.log_stat("noopt_loss", noopt_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (Q_joint_hat * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
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
        self.rest_mac.cuda()
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
