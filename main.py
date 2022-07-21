import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
# Scared是一个开源的Python软件包，利于配置，组织，记录和重现深度学习实验
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")  # Experiment 类是 Sacred 框架的核心类.
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

# 让debug的结果单独存
src_path = dirname(abspath(__file__))  # '/home/Qshaye/pymarl/debug_src'
results_path = os.path.join(dirname(src_path), "debug_results")
# if src_path.find('debug_src') >= 0:
#     results_path = os.path.join(dirname(src_path), "debug_results")
# else:
#     results_path = os.path.join(dirname(src_path), "results")
# os.path.join: 把目录和文件名合成一个路径
# __file__表示当前.py文件的路径
# os.dirname() 找到文件所在文件夹的路径 os.path.abspath() 表示当前文件绝对路径
# 合并起来就是找上级目录


@ex.main
def my_main(_run, _config, _log):
    # _config内容就是配置文件中的参数 _log是logger
    # 设置随机种子 从 run.py 开始运行
    config = config_copy(_config)  # 复制参数 ==
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


# 用于找到输入指定的配置文件 envs 和 algs
def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    # isinstance(a, b) 判断 a, b 是不是同一个类型
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=') + 1:].strip()
            break
    return result


if __name__ == '__main__':
    params = deepcopy(sys.argv)  # 读取运行时传入的参数 如 算法 地图

    # 把default.yaml中的 全部设置 加入到 config_dict
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")  # 根据输入“--env-config==sc2” 加载对应的yaml配置
    alg_config = _get_config(params, "--config", "algs")  # 同理，循环查找是否有输入对应的算法 在config文件中找到对应配置文件

    # 将所有配置参数更新/加入到 config_dict中
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # Add a configuration entry to this experiment. 用sacred记录所有的配置
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name'])
    file_obs_path = os.path.join(results_path, "sacred", map_name, algo_name)

    logger.info("Saving to FileStorageObserver in {}".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
    # 运行后会跳转到 @ex.main 修饰的函数内，一般放在文件的末尾
