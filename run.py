import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
# import wandb
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env


def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return 4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    # args <- config
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    #打印出所有设置的参数
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # 根据 所选算法和开始时间 设置一个token 例如 'qmix__2022-03-07_08-29-12'
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    # 程序正式开始运行
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    # 程序结束后打印信息
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # if dirname(abspath(__file__)).find('debug_src') >= 0:
    #     mywandb = wandb.init(project="MARL-{}-debug".format(args.env_args['map_name']), config=vars(args), name=unique_token)
    # else:
    #     mywandb = wandb.init(project="MARL-{}".format(args.env_args['map_name']), config=vars(args), name=unique_token)
    # 根据配置信息 创建 runner，qmix 为 episode_runner
    # 在runner初始化时会创建 smac环境
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    # 获取环境信息 stateShape obsShape nAction nAgent 等
    args.n_agents = env_info["n_agents"]
    # agent 个数会由于地图设定而不同
    args.n_actions = env_info["n_actions"]
    # 动作维度 不同地图会不一样
    args.state_shape = env_info["state_shape"]
    
    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)
        
    # Default/Base scheme
    # scheme 存了当前环境的整体基本信息
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 创建了replayBuffer 并且将数据都初始化为0(按要求的type和shape) <- __init__方法
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # 根据 args 选择 controller (qmix为 basicMAC)
    # mac 创建了agent 和 selector
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # setup里面的partial函数创建了一个函数newbatch()  创建episode_batch
    # runner.mac
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # learner 创建时会创建 mix 网络 设置需要优化的参数 创建optimizer 复制两个target网络
    # learner.mac and learner.target_mac
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    # 如果配置文件选择 使用cuda(默认为true)， 就把learner放到cuda上运作
    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    # 开始训练
    episode = 0
    last_test_T = -args.test_interval - 1  # 上一次test的时间 用于条件判断
    last_log_T = 0  # 上一次打印的时间
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    # 输出提示：my_main Beginning training for "t_max" timesteps
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # 主循环 t_env：所有用于train的时间片
    while runner.t_env <= args.t_max:

        with th.no_grad():
            # 1. 获取经验样本，run函数一次运行1个episode（episode_batch）
            episode_batch = runner.run(test_mode=False)  # 这里runner的reset又会创建一个容器self.batch来返回

            # 2. 把episode_batch个 episode的样本存入replay buffer
            buffer.insert_episode_batch(episode_batch)

        # 3. 如果buffer中的样本足够训练，则开始训练-learner
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)  # buffer里sample数据

            # 找出最长的episode的时间 裁剪数据
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)  # 传入episode以判断是否需要更新网络 env_t：是否打印log
            del episode_sample

        # 4. 判断是否进入测试
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            # 测试模式 不接收batch
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # 5. 判断是否保存模型
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        # 6. 更新当前episode数量
        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()  # 输出最近的一些信息
            last_log_T = runner.t_env
            # mywandb.log({
            #     'timesteps': runner.t_env,
            #     'return_mean': th.mean(th.tensor([float(x[1]) for x in logger.stats['return_mean'][-5:]])),
            #     'battle_won_mean': th.mean(th.tensor([float(x[1]) for x in logger.stats['battle_won_mean'][-5:]])),
            #     'test_return_mean': th.mean(th.tensor([float(x[1]) for x in logger.stats['test_return_mean'][-5:]])),
            #     'test_battle_won_mean': th.mean(th.tensor([float(x[1]) for x in logger.stats['test_battle_won_mean'][-5:]])),
            # })

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
