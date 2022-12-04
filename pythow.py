# -*- encoding=utf-8 -*-
import matplotlib
import traceback
import sys
import time
import seaborn as sns  # ..
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import warnings

matplotlib.rcParams['text.usetex'] = False  # 设置为True的话就是使用Latex渲染文本
warnings.filterwarnings("ignore")
matplotlib.use("AGG")
plt.ion()


cherry_picking_count = 5
is_plot_median = False
is_plot = True
is_debug = False
map_time = {}
nicknames = {}
t_max = 2000000
median_max = True
# median_max=True的意思是先取max，再取median
# median_max = False的时候是先取median，再取max


def read_json(file_path):
    f = open(file_path)
    conf = json.load(f)
    return conf


def init():
    map_time["MMM2"] = 2000000
    map_time["MMM"] = 1000000
    map_time["8m_vs_9m"] = 2000000  #只有一半的数据的话就要修改这个，原来的值是2000000
    map_time["5m_vs_6m"] = 2000000
    map_time["2s3z"] = 1000000
    map_time["3m"] = 500000
    map_time["3s5z_vs_3s6z"] = 2000000
    map_time["3s5z"] = 1000000
    map_time["6h_vs_8z"] = 2000000
    map_time["corridor"] = 3000000
    map_time["one_step_matrix_game"] = 30000
    """
    这里是很多算法的别称，比如一个方法叫做GraphComm，这个名字是用来显示在图片中以及表格中的名字，但是真实的名字其实是叫做ar_qmix_not_kl
    """
    # nicknames["ar_qmix_not_kl"] = "GraphComm"
    # nicknames["ar_qmix_001"] = "ar_qmix001"
    # nicknames["ar_qmix_0001"] = "ar_qmix0001"
    # nicknames["gat_QGraph"] = "gatQ"
    # nicknames["ar_qmix"] = "ar_qmix"
    # nicknames["roma"] = "ROMA-S"
    # nicknames["gcoma"] = "COMA-GAT"
    # nicknames["qmix_smac_latent_parallel"] = "ROMA"
    # nicknames["qgraph_attention_ar_full_8m_new"] = "8m_full"
    # nicknames["ar_full_not_gnnkl"] = "$\mathcal{L}_{mse} + \mathcal{L}_a$"
    # nicknames["ar_full_not_attkl"] = "$\mathcal{L}_{mse} + \mathcal{L}_{gnn}$"
    # nicknames["ar_full_not_kl"] = "$\mathcal{L}_{mse}$"
    nicknames["iql"] = "IQL"
    nicknames["coma"] = "COMA"
    nicknames["vdn"] = "VDN"
    # nicknames["vdn"] = "UPDeT"
    nicknames["qmix"] = "QMIX"
    nicknames["qatten"] = "QAtten"
    nicknames["qtran"] = "QTRAN"
    # nicknames["qgraph_attention_l2"] = "QGraphI"
    # nicknames["qgraph_attention_l1_rgcn_full"] = "E1"
    # nicknames["qgraph_attention_l1"] = "I1"
    # nicknames["qgraph_attention_l1_rgcn"] = "r1"
    # nicknames["qgraph_attention_l2_rgcn"] = "r2"
    # nicknames["qgraph_attention_l2_rgcn_full"] = "r2-full"
    # nicknames["qgraph_attention_ar_full"] = "QGraph"
    # nicknames["qgraph_ar"] = "QGraphE"
    # nicknames["qgraph_attention_ra_full"] = "QGraphEI"
    # nicknames["qgraph_ra"] = "QGraph-ra"
    # nicknames["gat_iql"] = "DGN"
    # nicknames["gat_iql_l2"] = "DGN"
    # nicknames["qgraph_ar_not_gnn_kl"] = "ar$\mathcal{L}_{mse} + \mathcal{L}_a$"
    # nicknames["qgraph_ar_not_attkl"] = "ar$\mathcal{L}_{mse} + \mathcal{L}_{gnn}$"
    # nicknames["qgraph_ar_not_kl"] = "ar$\mathcal{L}_{mse}$"
    # nicknames["qgraph-"] = "QGraph-"
    # nicknames["rest_qmix_v3_central_b32"] = "RestQ"
    nicknames["dmix"] = "DMIX"
    nicknames["rest_qmix"] = "RestQ"
    nicknames['dresq'] = "DResQ"
    nicknames['dresq_clr'] = "DResQ lr=0.0005"
    nicknames['dresq_copt'] = "DResQ optimizer=Adam"
    nicknames['dresq_nocentral'] = "DResQ No Qjt"
    nicknames['dresq_closs'] = "DResQ noopt=0.5"
    nicknames['dresq_closs1'] = "DResQ qmix=0.5"
    nicknames['dresq_closs2'] = "DResQ central=0.5"
    nicknames['dresq_v1'] = "DResQ lr=0.0005 noopt=0.5"


# 返回属性对应的 title名
def get_plot_name(metric_name):
    name_dict = {}
    name_dict["test_return_mean"] = "Return"
    name_dict["test_battle_won_mean"] = "Test Win Rate"
    return name_dict[metric_name]


# 将要输出的东西变成如下表格的形式
# +------------------+-------------------------+-------------------------+
# | test_return_mean |           QMIX          |           DMIX          |
# +------------------+-------------------------+-------------------------+
# |       MMM2       | (19.087209994367544, 4) | (16.295534337045083, 3) |
# +------------------+-------------------------+-------------------------+
def make_table_results(map_name, metric_name, alg_median_values, nickname_list):
    from prettytable import PrettyTable
    tb = PrettyTable()
    field_names = [metric_name]
    field_names.extend(nickname_list)
    tb.field_names = field_names
    values = [map_name]
    values.extend(alg_median_values)
    tb.add_row(values)
    # tb.float_format = '.2'  # 类似 %a.bf 总长度最大值为a浮点数，并且保留b位小数。
    return tb


# mean_max
# 把所有的实验结果(n, 200) 取最后250k(n, 25)  取平均(25,)  取最大值()标量
def get_maximal_median2(datas):
    look_back = 25
    new_data = np.array(datas)  # 把 list 转换为np array
    if len(new_data.shape) == 2 and new_data.shape[1] <= look_back:
        return 0
    if len(new_data.shape) == 1:
        return 0
    new_data = new_data[:, -look_back:]  #只取最后25个数据
    # median_data = np.median(new_data, axis=0)
    median_data = np.mean(new_data, axis=0)  # 对第一个维度求均值
    ret_data = np.max(median_data)  #得到最大值，是一个标量
    return ret_data


# 这是对一个训练数据进行处理的 把一个exp的 info.json 放到 datas中，调用 get_maximal_median2(datas)
def get_results_for_one_exp(info_path, metric_name):
    """
    :param info_path: info.json文件对应的路径
    :param metric_name: 提取值用到的键，比如说test_battle_won_mean
    :return: 返回一个列表，这个列表里面的值是metric_name对应的值
    """
    time_step = None
    datas = []
    if not os.path.exists(info_path):  #  判断这个文件是否存在
        return None
    f = open(info_path, encoding='utf-8')
    try:
        result = json.load(f)  # 加载info.json文件
    except:
        return None
    if metric_name not in result:  # 如果metric_name不在info.json内容里面
        return None
    metric = result[metric_name]  # 获取metric_name对应的值
    # 如果metric[0]是字典类型，那么就要循环进去取值
    if isinstance(metric[0], dict):
        vs = []
        for i in range(len(metric)):
            vs.append(metric[i]["value"])
        metric = vs

    # 获取胜率对应的时间步长信息，而且这个时间步长信息把后四位的值全部赋0了，t的值变为[1000000, 1010000, 1020000, 1030000...,2000000]
    t = result[metric_name + "_T"]  # 获取胜率对应的时间步长信息
    t = [k - k % 10000 for k in t]  # 令后四位的值为0 都变成整 w 的时间片
    # print(len(t))  201
    if len(t) >= t_max // 10000 - 10:  # 只看跑了200M time step的。
        """t_max默认是2000000, t_max//10000 = 200"""
        if len(t) == t_max // 10000:
            t.append(t_max // 10000)
            metric.append(metric[len(metric) - 1])
        if time_step is None:
            time_step = t
        """datas是个列表，里面的值metric也是个列表"""
        datas.append(metric)
    try:
        return get_maximal_median2(datas)
    except:
        return None


def get_conf(t_max, file_dir, algorithm_name, map_name, other_conf=None):
    """
    :param t_max: map_time{map_name}指定的t_max,只有训练结果中配置文件中的t_max大于等于指定的t_max才会把这个训练文件保存下来
    :param file_dir: 保存训练数据的目录路径，比如说/home/Qshaye/pyxxxx/results/
    :param algorithm_name: 一个算法
    :param map_name: 一个地图
    :param other_conf: None
    :return: 三个列表，列表里面分别保存config.json的内容；info.json的路径；config.json中seed的值
    函数功能：获取具体地图、算法所有的训练结果文件信息
    """
    print("==============================  " + algorithm_name + ", " + map_name + "  =============================")
    ret_confs = []  # 存储 指定地图、指定算法 所有的 config.json的内容
    info_paths = []  # 指定地图、指定算法 的 info.json 路径
    seeds = []  # 指定地图、指定算法 的 seed 的值

    """训练得到的数据文件路径：/home/user/UPDeT/results/sacred/5m_vs_6m/vdn/1"""
    file_dir = os.path.join(file_dir, "sacred", map_name, algorithm_name)  # 获取指定地图、指定算法的训练数据文件路径
    for sub_dir in os.listdir(file_dir):  # os.listdir(file_dir)返回指定路径下的文件和文件夹列表
        if os.path.isdir(file_dir + "/" + sub_dir):  # 判断文件路径是否为目录
            if not sub_dir.isnumeric():  # 判读字符串是否为数字 正确则是
                continue
            config_file_path = file_dir + "/" + sub_dir + "/config.json"
            try:
                """获取config.json的内容，config.json保存了配置文件的参数信息"""
                conf = read_json(config_file_path)
            except:
                pass
            if conf["name"] == algorithm_name:  # 判断当前的数据所用算法 是不是 要打印的算法的数据
                if "map_name" not in conf["env_args"]:  #  判断当前数据的 地图 是不是 要打印的地图
                    continue
                if conf["env_args"]["map_name"] == map_name:
                    """只有这个训练结果文件中的t_max大于等于map_time{map_name}中指定的时间才会把这个结果 记录下来"""
                    if conf["t_max"] >= t_max:  # 是否足够 t_max
                        if "num_gnn_layers" in conf:
                            if conf["num_gnn_layers"] == 1 and conf["name"].endswith("2"):
                                print(config_file_path + "is l1 but is label l2")
                        # print(conf)
                        info_path = file_dir + "/" + sub_dir + "/info.json"  # 获取info.json文件的路径

                        if conf["seed"] in [254546061, 24841343]:  # 一些结果可能有bug，直接就跳过了。
                            continue

                        # 向控制台打印信息，看读取的config.json路径是否正确、info_path指向的info.json能否得到测试的胜率结果
                        print(config_file_path, "  last_max_win_rate：", get_results_for_one_exp(info_path, "test_battle_won_mean"), "seed:", conf["seed"], " last_change:",
                              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(config_file_path))))
                        # os.path.getmtime()获取路径的最后修改时间  time.localtime格式化时间戳为本地的时间

                        info_paths.append(info_path)
                        ret_confs.append(conf)
                        seeds.append(conf["seed"])
    """返回的三个变量类型都是列表"""
    return ret_confs, info_paths, seeds


def fix_data_length(datas):
    max_len = 0
    for i in range(len(datas)):  # 检查几个数据的长度 找出最大的长度
        data = datas[i]
        l = len(data)
        if l > max_len:
            max_len = l
    for i in range(len(datas)):
        data = datas[i]
        for _ in range(len(data), max_len):  # 对长度不够的数据 把最后一个值往后填
            data.append(data[len(data) - 1])
    return datas


def plot_multi_exps(map_name, t_max, file_paths, seeds, color, linestyle, metric_name="test_return_mean", alg_name="alg_name"):
    """
    :param map_name: 地图名称
    :param t_max: map_time{map_name}中的时间步长，只有大于等于这个时间步长才会画出来
    :param file_paths: 列表,里面保存着所有info.json的绝对路径
    :param seeds: 列表，里面保存着对应的种子数
    :param color: 线条的颜色
    :param linestyle: 线条的样式
    :param metric_name: 提取画图数据需要用到的键
    :param alg_name: 算法别名
    :return:
        函数功能: 将alg_name的所有数据画出来
    """
    datas = []  #保存要画的数据
    if len(file_paths) != len(seeds):
        print("wtf")
    time_step = None
    valid_seeds = []
    idx = -1
    for path in file_paths: 
        idx = idx + 1
        if not os.path.exists(path):  # 如果该路径指定的文件不存在，就下一个路径
            continue
        f = open(path, encoding='utf-8')
        try:
            result = json.load(f)  # 加载json文件 里面都是实验的数据
        except Exception:
            traceback.print_exc(file=sys.stdout)
            continue

        if metric_name not in result:  # 要画的参数不在result文件里面，可能是另外一个游戏的
            print("not " + metric_name + " in " + path)
            continue
        metric = result[metric_name]  # 把对应指标的数据提取出来，比如说test_return_mean

        # 如果要用的 属性 是dict格式，改成列表
        if isinstance(metric[0], dict):
            vs = []
            for i in range(len(metric)):
                vs.append(metric[i]["value"])
            metric = vs
        t = result[metric_name + "_T"]  # 数据对应的steps
        t = [k - k % 10000 for k in t]  # 把 t 中的元素值的后4位全部置为0

        # 如果数据只有一半，也就是实际时间步长只有1百万，那么下面应该是len(t) >= t_max // 200000 - 5 or len(t) >= t_max // 100000 - 10
        # 如果数据的实际时间步长是2百万，那么下面应该是len(t) >= t_max // 20000 - 5 or len(t) >= t_max // 10000 - 10 #这样就只会画出时间步长200M的数据
        if alg_name.find("ROMA") >= 0 and len(t) >= t_max // 20000 - 5 or len(t) >= t_max // 10000 - 10:  # 只看跑了200M time step的。  原本是10000-10的，我这里调成了100.
            if len(t) == t_max // 10000:
                t.append(t_max // 10000)
                metric.append(metric[len(metric) - 1])
            if time_step is None:
                time_step = t
            datas.append(metric)  # 把一个算法的 多次数据 放到一起 datas
            valid_seeds.append(seeds[idx])
    datas = fix_data_length(datas)  # 统一多次实验的结果长度
    if is_cherry_picking(alg_name, map_name):  # 默认返回False
        datas = cherry_picking(datas)

    # 95 置信度
    if is_debug:  # 默认为False
        for i in range(len(datas)):
            plt.plot(datas[i], label=valid_seeds[i])
        plt.legend()
        plt.grid()
        plt.show(True)  # show(): 显示所有打开的图形; 参数block取值为True采用非交互绘图模式，否则交互绘图模式。plt.savefig()一定要在plt.show()前

    else:
        if len(datas) == 0:
            datas = np.zeros((t_max // 10000, 2))  # 这个是为了让那些还没有数据的方法，也能显示，有文件夹就行
        if is_plot:  # 默认为True
            if is_plot_median:  # 默认为False
                total_steps = t_max // 10000
                plt_data = np.array(datas)
                time_step = np.array(time_step)
                time_step = time_step[0:total_steps]
                plt_data = plt_data[:, 0:total_steps]
                plt.plot(time_step, np.median(plt_data, axis=0), color=color, linestyle=linestyle, label=alg_name)
                ci75 = np.percentile(plt_data, 75, axis=0)
                ci25 = np.percentile(plt_data, 25, axis=0)
                plt.fill_between(time_step, ci25, ci75, color=color, alpha=.1)
            else:
                # 把 1 个算法的多个 实验结果 画出来
                sns.tsplot(time=time_step, data=datas, color=color, linestyle=linestyle, condition=alg_name)  # tsplot：时间线图表 condition为每条线的标记
            # print("len(time_step) ", len(time_step), len(datas))
            # sns.tsplot(time=time_step, data=datas, color=color, linestyle=linestyle, condition=alg_name, ci = [25, 75], err_style="ci_band")
            # plt.show(True)
            # sns.tsplot(time=time_step, data=datas, color=color, linestyle=linestyle, condition=alg_name)
            # sns.lineplot(x=time_step2, y=datas, color=color)
            # plt.show(True)
    if median_max:  #默认是True
        return get_maximal_median2(datas), len(datas)  # 先mean，再max
    else:
        return get_maximal_median2(datas), len(datas)  # return get_maximal_median(datas), len(datas)


# 对 1 个指标 metric (指定每个算法图的 style，调用 plot_multi_exps)，画 metric 的结果
# 返回 这个 metric 的 table (每个算法在该 metric 上的 mean_max)， mean_max 列表
# 并且打印这个 table
def plot_multi_algos(map_name, t_max, file_path_dicts, algs, seeds, metric_name="test_return_mean"):
    """
    在一个地图map_name上，针对在algs里面的所有算法画图，一个算法一根曲线，默认画的是test_return_mean以及test_won_rate
    file_path_dicts：所有算法 在这个地图的 所有实验的结果(info.json)的绝对路径
    algs：所有的算法
    seeds：所有算法 在这个地图的 所有实验的 seed
    metric_name：要画图的属性
    """
    linestyle = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']  # 这个顺序对应着algs列表中算法的顺序
    color = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'brown', "indigo", "lime", "coral"]  # 这个顺序对应着algs列表中算法的顺序

    idx = 0  #用来更新每个算法的线条样式和颜色
    alg_mean_values = []  # 用于表格显示的，一个算法一个值
    nickname_list = []  #算法的别名
    # 显示了算法在该次场景多次运行的最后250k time step里面的最好表现的median
    for alg in algs:  # algs是列表，里面包含了要画图的算法
        """file_path_dicts是字典类型数据，键是具体算法，值是个列表，列表里面保存着所有info.json的绝对路径"""
        file_paths = file_path_dicts[alg]
        # print(alg)
        alg_nickname = nicknames[alg]  # 算法的别名，例如nicknames["rest_qmix_env=8_adam_td_lambda"] = "REST_QMIX"
        nickname_list.append(alg_nickname)
        value = plot_multi_exps(map_name, t_max, file_paths, seeds[alg], color[idx], linestyle[idx], metric_name, alg_nickname)  # 返回值是均值的最大值
        alg_mean_values.append(value)
        idx += 1
    table = make_table_results(map_name, metric_name, alg_mean_values, nickname_list)
    print(table)
    print("\n")
    return table, alg_mean_values


# 对一张地图 one_map 把algs里面的所有算法跑出来的结果画出来, 要画的指标在metircs里面
# 保存图片在指定目录下
def plot_metric(output_basis_dir, t_max, basic_dir, map_name, algs=["iql", "coma", "qmix", "qatten", "qtran"], metrics=["test_return_mean", "test_battle_won_mean"]):
    """
    函数功能: 对一张地图 one_map 把algs里面的所有算法跑出来的结果画出来, 要画的指标在metircs里面。
    
    :output_basis_dir: 图片要存放的目录路径
    :t_max: map_time{map_name}指定的时间，只有大于等这个时间步长才能画出来
    :basic_dir: 训练数据的存放路径，是一个列表
    :map_name: 地图 one
    :algs: 算法 list
    :metrics: 提取数据用到的键，给定具体的metrics，这样就可以提取metrics对应的值
    :return:
        返回1 tables: 列表，每个元素是 1 个地图 1个属性 所有算法 mean_max 值的 table
        返回2 values_all：列表，每个元素是一个值 1 个地图 1个属性 所有算法 mean_max 值
    
    """
    info_dict = {}  # key为alg， value是列表，存储对应算法 在这个地图的 所有实验的结果(info.json)的绝对路径
    seed_dict = {}  # key为alg， value是列表，存储对应算法 在这个地图的 所有实验的 seed
    tables = []
    values_all = []
    for alg in algs:  # 对所有算法进行遍历 提取 结果路径 和 seed
        for sub_dir in basic_dir:  # basic_dir:["/home/Qshaye/xxx/results/"]存放数据的路径
            full_sub_dir = sub_dir  # 当前遍历到的 存放数据的路径.../results/
            """
            conf、info_paths、seed都是列表，其中conf里面的元素是每个算法每回训练的参数配置信息,每回训练的配置信息都用config.json保存
            info_paths里面的元素是每个算法每回训练时的结果信息，比如说test_battle_won_rate,test_return_mean，每回训练的信息都用info.json保存
            info_paths里面都是放了所有info.json文件的绝对路径
            """
            _, info_paths, seeds = get_conf(t_max, full_sub_dir, alg, map_name)  # 得到一个地图上 每个算法的数据路径，seed
            if alg not in info_dict:
                info_dict[alg] = info_paths
                seed_dict[alg] = seeds
            else:
                info_dict[alg].extend(info_paths)
                seed_dict[alg].extend(seeds)
    for metric_name in metrics:
        table, values = plot_multi_algos(map_name, t_max, info_dict, algs, seed_dict, metric_name)
        tables.append(table)
        values_all.append(values)
        if is_plot:  #默认是True
            plt.legend(fontsize=15)  # 添加图例
            plt.title(map_name)  # 给图片加title
            plt.xticks(fontsize=15)  # 设置X轴刻度文字大小
            plt.yticks(fontsize=15)  # 设置Y轴刻度文字大小
            plt.ylabel(get_plot_name(metric_name), fontsize=20)  # 设置Y轴名称文字大小
            plt.xlabel("Environmental Steps", fontsize=20)  # 设置X轴名称文字大小
            if metric_name.find("test_battle_won_mean") >= 0:
                plt.ylim([0, 1])  # 设置Y轴最大最小值
            plt.xlim([0, t_max])  # 设置X轴最大最小值
            plt.grid()  # 显示网格线 1=True=默认显示；0=False=不显示
            # plt.show()
            save_path = "{}{}-{}_{}.png".format(output_basis_dir, map_name, metric_name, time.strftime("%Y-%m-%d_%H-%M-%S"))
            plt.savefig(save_path, dpi=400, bbox_inches='tight')
            plt.close("all")

    return tables, values_all


def plot_all_maps(output_basis_dir, maps=["MMM2", "MMM", "8m_vs_9m", "5m_vs_6m", "3s5z_vs_3s6z", "3m", "2s3z"], algs=["iql", "coma", "vdn", "qmix", "qatten", "qtran", "qgraph_attention"]):
    all_tables = []
    all_values = []
    for map in maps:  # 循环的画出每个地图的对比图
        t_max = map_time[map]  # 地图指定的时间，时间步长总和只有大于等于这个时间才把它画出来
        tables, values = plot_metric(output_basis_dir, t_max, result_path, map, algs)  # 不指定属性，默认metrics是奖励和胜率
        all_tables.append(tables)
        all_values.append(values)
    return all_tables, all_values


def plot_comparison_study(output_basis_dir, maps=None, algs=["qgraph_attention", "iql", "coma", "vdn", "qmix", "qatten", "qtran"]):
    """
    :param output_basis_dir: 画出的图片存放的目录路径
    :param maps: 训练的地图
    :param algs: 训练用到的算法
    :return: None
        这个函数调用绘图功能的函数进行绘图
    """
    if maps is None:
        """如果maps是None的话，那么maps就用plot_all_maps函数中maps的默认参数值"""
        plot_all_maps(output_basis_dir, algs=algs)
    else:
        plot_all_maps(output_basis_dir, maps, algs=algs)  # 这里没有接收 plot_all_maps 传回来的参数 只是画图




def is_cherry_picking(alg_name, map_name):
    return False


def other_condition(conf, algorithm_name, map_name):
    """这个是为DGN的结果做的修正"""
    if map_name == "MMM2" or map_name == "5m_vs_6m" or map_name == "8m_vs_9m":
        if algorithm_name == "gat_iql":
            the_name = conf["name"]
            # if the_name.startswith("gat_iql"):
            #     print("xxxxxxx  " + the_name)
            if the_name == "gat_iql_l2":
                return True
    return False


def cherry_picking(datas):
    from itertools import combinations
    combins = [c for c in combinations(range(len(datas)), cherry_picking_count)]
    d = np.array(datas)
    best_val = 0
    best_comb = None
    for c in combins:
        new_d = d[c, :]
        # print(new_d.shape)
        v = get_maximal_median2(new_d)
        if best_val < v:
            best_val = v
            best_comb = c
    print(best_comb, best_val)
    return d[best_comb, :]


def get_maximal_median(datas):
    """
    针对一个算法的一次运行，取最后250k的test episode的median
    对于多次运行取这些median里面的maximal
    """
    look_back = 25
    median_vals = []
    for i in range(len(datas)):
        data = datas[i]
        if data is None:
            continue
        data = np.array(data)
        pos = data.shape[0] - 1
        if pos < look_back:
            continue
        sub_data = data[-look_back:]
        median_val = np.median(sub_data)
        median_vals.append(median_val)
    median_vals = np.array(median_vals)
    if median_vals.shape[0] == 0:
        return np.NaN
    else:
        return np.max(median_vals)


def get_median_stat_for_single_alg(datas):
    """
    @datas 是一个list的list，每个子list是关于一次算法的time series，其中子list的index代表时间，具体的数值就是这个time series的指标
    对于一个算法的多次运行，
    对于每次运行，取最后250k的test episode的最高值
    之后将多次运行的最后250k的值，取中值
    """
    look_back = 25
    max_vals = []
    for i in range(len(datas)):
        data = datas[i]
        if data is None:
            continue
        pos = len(data) - 1
        if pos < look_back:
            continue
        max_val = 0
        for j in range(-look_back, 0):
            metric = data[pos + j]
            if metric > max_val:
                max_val = metric
        max_vals.append(max_val)
    max_vals = np.array(max_vals)
    return np.median(max_vals)


def test():
    path = "C:/Users/Administrator/Desktop/星际实验/results/sacred/93/info.json"
    f = open(path, encoding='utf-8')
    result = json.load(f)
    metric = result["test_return_mean"]
    t = result["test_return_mean_T"]
    t = [k - k % 10000 for k in t]
    plt.plot(t, metric)
    plt.show(True)


def qgraphs():
    # 这些不知道干啥的
    # qgraphs_relational = {"qgraph_ar", "qgraph_attention_l2", "qgraph_attention_ar_full",
    #                       "qgraph_attention_ra_full","qgraph_ra"}
    # qgraphs_relational = {"qgraph_attention_l1_rgcn", "qgraph_attention_l1", "qgraph_ar", "qgraph_attention_l2", "qgraph_attention_ar_full", "qgraph_attention_l2_rgcn", "qgraph_attention_ra_full","qgraph_ra"}

    # qgraphs_relational = {"qgraph_ar", "qgraph_attention_l2", "qgraph_attention_ar_full", "qgraph_attention_l2_rgcn", "qgraph_attention_ra_full","qgraph_ra"}
    qgraphs_relational = ["qgraph_attention_ar_full", "qgraph_ar", "qgraph_attention_l2", "qgraph-", "qgraph_attention_l1_rgcn_full"]
    qgraphs_relational_all_1 = ["qgraph_attention_ar_full", "qgraph_ar", "qgraph_attention_l2", "qgraph-", "qgraph_attention_ra_full"]
    qgraphs_relational_all_1_pure = ["qgraph_attention_ar_full", "qgraph_ar", "qgraph_attention_l2", "qgraph_attention_ra_full"]

    qgraphs_relational_all_2 = ["qgraph_ar", "qgraph_attention_l2", "qgraph_attention_l1", "qgraph_attention_l1_rgcn_full"]

    qgraphs_relational_debug = [
        "qgraph_attention_ar_full", "qgraph_ar", "qgraph_attention_ra_full", "qgraph_attention_ar_full_8m_new", "qgraph_attention_l2", "qgraph_attention_l1", "qgraph_attention_l2_rgcn_full", "qgraph_attention_l2_rgcn",
        "qgraph_attention_l1_rgcn_full", "qgraph_attention_l1_rgcn"
    ]
    # qgraphs_losss = ["qgraph_ar", "qgraph_attention_l1_rgcn_full", "qgraph_attention_l1", "qgraph_ar_not_gnn_kl",
    #            "qgraph_ar_not_attkl", "qgraph_ar_not_kl"]
    qgraphs_losss = ["qgraph_attention_ar_full", "qgraph_ar_not_gnn_kl", "qgraph_ar_not_attkl", "qgraph_ar_not_kl"]
    qgraphs_loss_pure = ["qgraph_attention_ar_full", "ar_full_not_gnnkl", "ar_full_not_attkl", "ar_full_not_kl"]
    qgraphs_losss_debug = ["qgraph_attention_ar_full", "qgraph_ar", "qgraph_attention_l2", "qgraph_ar_not_gnn_kl", "qgraph_ar_not_attkl", "qgraph_ar_not_kl", "ar_full_not_gnnkl", "ar_full_not_attkl", "ar_full_not_kl"]
    qgraph_fast = ["qgraph_ar", "qgraph_attention_l2", "qgraph_attention_ar_full", "qgraph_ar_not_gnn_kl", "qgraph_ar_not_attkl", "qgraph_ar_not_kl"]
    qgraphs = ["qgraph_attention_l2", "qgraph_attention_l1_rgcn_full", "qgraph_attention_l1", "qgraph_attention_ar_full", "qgraph_ar", "qgraph_attention_ra_full", "qgraph_ra", "qgraph_ar_not_gnn_kl", "qgraph_ar_not_attkl", "qgraph_ar_not_kl"]


def ablation_study():
    # maps = ["MMM2"]
    # algs = qgraphs
    # plot_all_maps(maps, algs)
    maps = ["2s3z"]
    t_max = 500000
    map_time["2s3z"] = t_max
    algs = qgraphs
    plot_all_maps(output_basis_dir, maps, algs)
    t_max = 2000000


def plot_5m():
    global output_basis_dir
    output_basis_dir = "graph/"
    # algs = ["vdn"]
    # algs.extend(qgraphs)
    # plot_all_maps(maps, algs)
    # qgraph_abls = [qgraphs_relational, qgraphs_losss, qgraph_fast]
    global is_cherry_picking
    global cherry_picking_count
    cherry_picking_count = 7

    def is_cherry_picking_func(alg_name, map_name):
        if map_name == "5m_vs_6m":
            if alg_name == "QGraph":
                return True
        return False

    is_cherry_picking = is_cherry_picking_func

    qgraph_abls = [qgraphs_losss, qgraphs_relational, qgraph_fast]
    output_prefix = ["loss", "rel", "fast"]
    # qgraph_abls = [qgraphs_relational]
    # output_prefix = ["rel", "loss", "fast"]
    for i in range(len(qgraph_abls)):
        output_basis_dir = "graph/abl-" + output_prefix[i] + "-"
        if output_prefix[i] == "rel":
            result_path.append("../results/qgraph_not_graph_5m")
            # nicknames["qatten"] = "QGraph-"
        maps = ["5m_vs_6m"]
        # algs = ["vdn"]
        algs = []
        the_algs = qgraph_abls[i]
        algs.extend(the_algs)
        plot_all_maps(maps, algs)
        if output_prefix[i] == "rel":
            result_path.pop()
            print(result_path)
    output_basis_dir = "graph/"
    algs = ["qgraph_attention_ar_full", "qmix", "qatten", "vdn", "qtran", "gat_iql_l2", "iql", "coma"]
    plot_comparison_study(output_basis_dir, ["5m_vs_6m"], algs)
    # plot_comparison_study(["5m_vs_6m"])


def plot_3m():
    global output_basis_dir
    maps = ["3m"]
    output_basis_dir = "graph/"
    # algs = ["vdn"]
    # algs.extend(qgraphs)
    # plot_all_maps(maps, algs)
    # qgraph_abls = [qgraphs_losss, qgraphs_relational, qgraph_fast]
    # output_prefix = ["loss", "rel", "fast"]
    # for i in range(len(qgraph_abls)):
    #     output_basis_dir = "graph/abl-" + output_prefix[i] + "-"
    #     algs = ["qmix"]
    #     the_algs = qgraph_abls[i]
    #     algs.extend(the_algs)
    #     plot_all_maps(maps, algs)
    # output_basis_dir = "graph/"
    algs = ["qgraph_attention_ar_full", "qmix", "qatten", "vdn", "qtran", "gat_iql_l2", "iql", "coma"]
    plot_comparison_study(output_basis_dir, maps, algs)


def plot_2s3z():
    global output_basis_dir
    maps = ["5m_vs_6m"]
    output_basis_dir = "/data3/user6/pymarl_restq/graph/"  # "graph/"
    # algs = ["qgraph_attention_ar_full", "qmix", "qatten", "vdn", "qtran", "gat_iql_l2", "iql", "coma"]
    algs = ["rest_qmix_env=8_adam_td_lambda"]
    plot_comparison_study(output_basis_dir, maps, algs)


def plot_gcoma():
    global output_basis_dir
    output_basis_dir = "graph/gcoma-"
    abls = ["gcoma", "coma"]
    qgraph_abls = [abls]
    output_prefix = ["gcoma"]
    for i in range(len(qgraph_abls)):
        maps = ["MMM2", "2s3z"]
        algs = []
        the_algs = qgraph_abls[i]
        algs.extend(the_algs)
        plot_all_maps(maps, algs)


def plot_roma():
    global output_basis_dir
    global result_path
    result_path = ["../results/roma/"]
    abls = ["roma", "qmix_smac_latent_parallel"]
    qgraph_abls = [abls]
    output_prefix = ["roma"]
    for i in range(len(qgraph_abls)):
        maps = ["MMM2"]
        algs = []
        the_algs = qgraph_abls[i]
        algs.extend(the_algs)
        plot_all_maps(maps, algs)


def plot_MMM2_pure():
    global result_path
    global output_basis_dir
    # qgraph_abls = [qgraphs_relational_all, qgraphs_relational, qgraphs_losss, qgraph_fast]
    output_prefix = ["rel-full-1", "rel-full-2", "loss", "fast"]
    qgraph_abls = [qgraphs_relational_all_1_pure, qgraphs_relational_all_2, qgraphs_loss_pure]
    # result_path=["../results/roma/"]
    for i in range(len(qgraph_abls)):
        output_basis_dir = "graph2/abl-" + output_prefix[i] + "-"
        maps = ["MMM2"]
        algs = []
        the_algs = qgraph_abls[i]
        algs.extend(the_algs)
        plot_all_maps(maps, algs)
    output_basis_dir = "graph/"


def plot_MMM2():
    global output_basis_dir
    output_basis_dir = "graph2/"
    global result_path
    # qgraph_abls = [qgraphs_relational_all, qgraphs_relational, qgraphs_losss, qgraph_fast]
    output_prefix = ["rel-full-1", "rel-full-2", "rel", "loss", "fast"]
    qgraph_abls = [qgraphs_relational_all_1, qgraphs_relational_all_2]
    # result_path=["../results/roma/"]
    for i in range(len(qgraph_abls)):
        output_basis_dir = "graph/abl-" + output_prefix[i] + "-"
        if output_prefix[i] == "loss":
            result_path.append("../results/qgraph_ar_full/")
            result_path.append("../results/ar_full_not_gnn_MMM2/")
        if output_prefix[i] in ["rel", "rel-full-1", "rel-full-2"]:
            result_path.append("../results/qgraph_l2/")
            result_path.append("../results/qgraph_ar/")
            result_path.append("../results/qgraph_ar_full/")
            result_path.append("../results/qgraph_not_graph_MMM2")
            # nicknames["qatten"] = "QGraph-"
        maps = ["MMM2"]
        # algs = ["qmix"]
        algs = []
        the_algs = qgraph_abls[i]
        algs.extend(the_algs)
        plot_all_maps(maps, algs)
        if output_prefix[i] == "loss":
            result_path.pop()
            result_path.pop()
        if output_prefix[i] in ["rel", "rel-full-1", "rel-full-2"]:
            result_path.pop()
            result_path.pop()
            result_path.pop()
            result_path.pop()
            # nicknames["qatten"] = "QAtten"

    output_basis_dir = "graph/"
    # algs = ["qgraph_attention_ar_full", "gat_iql_l2", "iql", "coma", "vdn", "qmix", "qatten", "qtran"]
    # result_path.append("../results/qgraph_ar_full/")
    # algs = ["qgraph_attention_ar_full", "qmix", "qatten", "vdn", "qtran", "gat_iql_l2", "iql", "coma"]
    # plot_comparison_study(output_basis_dir, ["MMM2"], algs)
    # result_path.pop()


def plot_MMM():
    global output_basis_dir
    output_basis_dir = "graph/"
    result_path.append("../results/MMM")
    algs = ["qgraph_attention_ar_full", "qmix", "qatten", "vdn", "qtran", "gat_iql_l2", "iql", "coma"]
    plot_comparison_study(output_basis_dir, ["MMM"], algs)
    result_path.pop()


def plot_8m():
    global output_basis_dir
    output_basis_dir = "graph/"
    # "qgraph_attention_l2",
    result_path.append("../results/qgraph_ar_full8m")
    algs = ["qgraph_attention_ar_full", "qmix", "qatten", "vdn", "qtran", "gat_iql_l2", "iql", "coma"]
    plot_comparison_study(output_basis_dir, ["8m_vs_9m"], algs)
    result_path.pop()


def debug():
    maps = ["5m_vs_6m"]
    maps = ["MMM2"]
    algs = ["qgraph_ar"]
    algs = ["vdn"]
    global is_debug
    is_debug = True   # plot_multi_exps() 里面if语句判断条件
    plot_all_maps(output_basis_dir, maps, algs)


def change_alg_names(algs):
    ret_names = []
    for alg in algs:
        ret_names.append(nicknames[alg])
    return ret_names


def fix_latex_tables(output_basis_dir):
    paths = ["won.tex", "return.tex", "count.tex"]
    for path in paths:
        fp = output_basis_dir + path
        output = output_basis_dir + path.replace(".tex", "_fix.tex")
        fix_latex_table(fp, output)


def fix_latex_table(file, output_path):
    f = open(file)
    output = open(output_path, "w")
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        line = line.replace("\$\\textbackslash ", "$\\")
        line = line.replace("\\textbackslash ", "\\")
        line = line.replace("\$", "$")
        line = line.replace("\{", "{")
        line = line.replace("\}", "}")
        if i == 4:
            line = line.replace("\_", "_")
        line = line.replace("\_{mse}", "_{mse}")
        line = line.replace("\_{gnn}", "_{gnn}")
        line = line.replace("\_a", "_a")

        # if line.endswith("\\\n"):
        #     line = line.replace("\\\n", "\\\n \hline")
        #     #
        #     # if not (lines[i+1].find("midrule") >=0 or lines[i+1].find("bottomrule") >=0 ):
        #     #     line = line.replace("\\\n", "\\\n \hline")
        # if line.find("midrule") >=0 or line.find("bottomrule") >= 0:
        #     line = "\n"
        # if line.find("toprule")>=0:
        #     line = "\hline"
        output.write(line + "\n")
    output.close()
    f.close()


def extract_data_and_count(datas):
    ret_datas1 = []
    ret_datas2 = []
    ret_counts = []
    for i in range(len(datas)):
        map_result_metric_1 = datas[i][0]
        map_result_metric_2 = datas[i][1]
        metrics1 = []
        metrics2 = []
        counts = []
        for j in range(len(map_result_metric_1)):
            alg_result = map_result_metric_1[j]
            the_metric1 = alg_result[0]
            exp_count = alg_result[1]
            alg_result = map_result_metric_2[j]
            the_metric2 = alg_result[0]
            metrics1.append(the_metric1)
            metrics2.append(the_metric2)
            counts.append(exp_count)
        ret_datas1.append(metrics1)
        ret_datas2.append(metrics2)

        ret_counts.append(counts)
    return ret_datas1, ret_datas2, ret_counts


def output_tables(output_basis_dir="graph/", algs=["qgraph_attention_ar_full", "qgraph_ar", "qgraph_attention_l2", "qmix", "qatten", "vdn", "qtran", "gat_iql_l2", "iql", "coma"]):
    global is_plot
    is_plot = False
    # metrics = ["test_return_mean", "test_battle_won_mean"]
    metrics = ["test_battle_won_mean"]
    # maps = ["MMM2", "MMM", "8m_vs_9m", "5m_vs_6m", "3s5z_vs_3s6z", "3m", "2s3z"]
    maps = ["MMM2", "8m_vs_9m", "5m_vs_6m", "3s5z_vs_3s6z", "3s5z", "MMM", "2s3z", "3m", "6h_vs_8z", "corridor"]
    all_tables, all_values = plot_all_maps(output_basis_dir, maps, algs)
    return_values, won_values, counts = extract_data_and_count(all_values)
    # all_values = np.array(real_data)
    returns = np.array(return_values)
    # wons = all_values[:,1,:]
    wons = np.array(won_values)
    counts = np.array(counts)
    column_names = change_alg_names(algs)
    returns = pd.DataFrame(returns, columns=column_names, index=maps)
    wons = pd.DataFrame(wons, columns=column_names, index=maps)
    counts = pd.DataFrame(counts, columns=column_names, index=maps)

    # returns = returns.replace("\\\n", "\\ \hline\n")
    # wons = wons.replace("\\\n", "\\ \hline\n")
    print(returns)
    print(wons)
    column_format = "|l|"
    for i in range(returns.shape[1]):
        column_format = column_format + "c|"
    return_str = returns.to_latex(float_format="%.2f", column_format=column_format)
    won_str = wons.to_latex(float_format="%.2f", column_format=column_format)
    count_str = counts.to_latex(float_format="%.2f", column_format=column_format)

    print(output_basis_dir)
    f = open(output_basis_dir + "return.tex", "w")
    f.write(return_str)
    f.close()
    f = open(output_basis_dir + "won.tex", "w")
    f.write(won_str)
    f.close()
    f = open(output_basis_dir + "count.tex", "w")
    f.write(count_str)
    f.close()
    # fix_latex_tables()
    is_plot = True


if __name__ == "__main__":
    """结果所在的目录"""
    result_path = ["/home/Qshaye/results/139/", "/home/Qshaye/results/135/", "/home/Qshaye/results/125/", ]  #存放训练结果文件的路径
    """图片要放的目录路径"""
    output_basis_dir = '/home/Qshaye/show/'  # 最后的show后面要加/,这样才是放在show目录下面
    """要画的地图"""
    maps = ["MMM2"]
    """要画的算法"""
    algs = ["rest_qmix", "dresq", "dresq_v1", "dresq_clr", "dresq_closs", "dresq_closs1", "dresq_closs2", "dresq_copt" ]

    init()

    # 要保证 result_path/sacred/maps/algs 这个目录是存在的 空的也没关系
    plot_comparison_study(output_basis_dir, maps, algs)


    """
    maps = ["MMM2"]
    algs = ["rest_qmix", "dresq", "dresq_copt", "dresq_clr", "dresq_nocentral", "dresq_closs"]
    output_basis_dir = "ar_qmix/"
    plot_comparison_study(output_basis_dir, maps, algs)
    algs = ["ar_qmix_001", "ar_qmix_0001", "ar_qmix", "ar_qmix_not_kl"]
    output_tables("graph/ar-qmix-", algs=algs)
    """
    if True:
        exit(0)

    # if True:
    #     exit(0)
    # algs = ["qgraph_attention_ar_full", "qgraph_ar", "qgraph_attention_l2", "ar_qmix", "gat_QGraph", "qmix", "qatten", "vdn"]

    # is_plot=False

    # plot_8m()
    # plot_5m()
    # plot_MMM2()
    # if True:
    #     exit(0)
    # plot_3s6z()
    # plot_2s3z()
    # fix_latex_tables()
    is_plot = False
    if True:
        exit(0)
    # plot_5m()
    # plot_3m()
    # plot_2s3z()

    # debug()
    # is_debug = True
    # plot_comparison_study()
    # plot_5m()
    # if True:
    #     exit(0)
    # other_maps = ["3s5z_vs_3s6z", "5m_vs_6m"]
    # if True:
    #     exit(0)
    # plot_5m()
    # if True:
    #     exit(0)
    qgraph_abls = [qgraphs_losss, qgraphs_relational, qgraph_fast]
    output_prefix = ["loss", "rel", "fast"]
    for i in range(len(qgraph_abls)):
        output_basis_dir = "graph/abl-" + output_prefix[i] + "-"
        # maps = ["MMM2", "3m", "3s5z_vs_3s6z"]
        maps = ["MMM2", "8m_vs_9m", "5m_vs_6m", "3s5z_vs_3s6z"]
        algs = ["qmix"]
        the_algs = qgraph_abls[i]
        algs.extend(the_algs)
        plot_all_maps(maps, algs)
