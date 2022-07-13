from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        # runner运行的环境数，runner.batch_size (默认为1)
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        # runner初始化时会将self.env设置为sc2环境实例
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit  # smac环境返回 episode_limit:120
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        # partial() 是被用作 “冻结” 某些函数的参数或者关键字参数，同时会生成一个带有新标签的对象
        # self.new_batch是一个固定某些参数的EpisodeBatch类
        # def __init__(self, scheme, groups, batch_size, max_seq_length,
        #            data=None, preprocess=None, device="cpu"):
        # 之前至少要传 scheme, groups, batch_size, max_seq_length 三个参数， 现在这三个都可以不传了
        # 之前缺省 device="cpu"， 现在有了args之后，缺省值被partial覆盖
        # 只需要调用 self.new_batch（） 就可以创建一个
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1, preprocess=preprocess, device=self.args.device)
        # runner的batch_size是
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()  # 创建了一个EpisodeBatch类对象 初始数据都为0 tensor
        self.env.reset()  # 会打印信息：开始游戏
        self.t = 0

    def run(self, test_mode=False):
        # 关键函数
        # 收集一个episode的样本数据

        self.reset()  # 初始化容器batch和env self.t置0

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)  # 每次episode开始，要给mac里的RNNAgent一个初始的h0

        # 运行一个episode
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],  # (120,)
                "avail_actions": [self.env.get_avail_actions()],  # list: 5 * 28 * 11
                "obs": [self.env.get_obs()]  # list: 5 * (80, )
            }

            # 调用EpisodeBatch类中定义的函数update
            # 作用：把当前时间片的 s, avail_a, o更新 batch中 (batch创建时初始化为0)
            self.batch.update(pre_transition_data, ts=self.t)  # 把当前时间片 t 传进去建立slice，可以按顺序放入batch中

            # 把exp交给agents的网络选择动作
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # 与环境交互得到reward
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward, )],
                "terminated": [(terminated != env_info.get("episode_limit", False), )],
            }

            self.batch.update(post_transition_data, ts=self.t)  # 把当前时间片的 a, r, done更新在batch中

            self.t += 1  # 每次一个时间片
        # episode 结束

        last_data = {"state": [self.env.get_state()], "avail_actions": [self.env.get_avail_actions()], "obs": [self.env.get_obs()]}
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)  # 现在self.batch中是一整个episode的转移数据
        ###
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns

        log_prefix = "test_" if test_mode else ""
        # 函数dict1.update(dict2) 把dict2中的内容加入到dict1中
        cur_stats.update(  # 更新三个属性：'battle_won', 'dead_allies', 'dead_enemies'
            {k: cur_stats.get(k, 0) + env_info.get(k, 0)
             for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)  # 已经运行的episode数目
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)  # 运行的时间(目前所有的episode)

        if not test_mode:  # t_env是所有训练所用的时间加和
            self.t_env += self.t

        cur_returns.append(episode_return)  # cur_returns记录每一个episode的奖励
        ###
        # 根据时间打印
        # 1.测试模式 & 达到要求的episode数目  // test_nepisode: 20 # Number of episodes to test for
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        # 2. 与上次打印train信息的时间 >= runner_log_interval: 2000 # 每2000timestep打印 runner stats
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            # hasattr(ob, A) 判断ob是否包含A属性 T/F
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
