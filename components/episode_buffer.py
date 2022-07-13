import torch as th
import numpy as np
from types import SimpleNamespace as SN

# 定义了EpisodeBatch类和ReplayBuffer类


class EpisodeBatch:

    def __init__(
            self,
            scheme,
            groups,  # groups = { "agents": args.n_agents}
            batch_size,  # batch -> 一个episode数据
            max_seq_length,  # 每个exp最长的t
            data=None,
            preprocess=None,
            device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            # 创建0初值数据
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        # 根据scheme中的信息初始化数据

        # 对preprocess中每一个元素处理
        # 把preprocess中的元素处理后放入scheme中
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]  # k-> new_k: 'actions'->'actions_onehot'
                transforms = preprocess[k][1]  # OneHot类

                # 在scheme中找到要预处理的元素的信息
                vshape = self.scheme[k]["vshape"]  # (1,)
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                # 给scheme中加入了新的元素 new_k(actions_onehot)， 并设定其正确的属性
                self.scheme[new_k] = {"vshape": vshape, "dtype": dtype}
                # 把 scheme[k] 的属性 添加到scheme[new_k]
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        # 给scheme添加元素 "filled"  当前时间片的exp的'filled'=1表示有数据
        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {
                "vshape": (1, ),
                "dtype": th.long
            },
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]

            # dict.get(key, default=0) 在字典中查找key的值，不存在返回0
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):  # 确保vshape都是(number, )格式
                vshape = (vshape, )

            # 如果是智能体的属性，group=1,将增加一个维度(n_agent)
            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:  # 这个属性是什么
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)

            # 根据scheme中元素的shape和type创建初始数据 初值0
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    # 用传入的data更新self.data
    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        # 两个slice组合，第一个处理维度1(选择batch)，第二个处理维度2(选择时间片)
        for k, v in data.items():

            if k in self.data.transition_data:

                target = self.data.transition_data

                # target["filled"]置为1 标志有效数据
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices

            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]

            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            # 把待更新数据变为所需格式
            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)

            # 更新内容
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            # 如果要更新的是 action，还要更新action_onehot
            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):  # 不太理解怎么检查的
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):

        if isinstance(item, str):  # 取对应属性 str 的值
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError

        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]] for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret

        # 传入的item为slices或下标 (将self.data中的数据裁剪需要的部分)
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()  # new_data只有两个dict: transition_data & episode_data = {}
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)  # 需要返回的batch_Size
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)  # 序列长度

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)  # data = new_data
            return ret

    def _get_num_items(self, indexing_item, max_size):  # 返回number

        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)

        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            # slice.indices( max_l) 返回slice的 start, stop(<=max_l), step 三个参数，
            return 1 + (_range[1] - _range[0] - 1) // _range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice) or isinstance(items, int) or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            # TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item + 1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    # 计算每一个batch的数据量，再max。 [32, 121, 1] -> [32, 1]
    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    # _repr__函数用于打印信息
    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size, self.max_seq_length, self.scheme.keys(), self.groups.keys())


class ReplayBuffer(EpisodeBatch):

    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # 可以存多少个episode采样的batch
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):

        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(
                ep_batch.data.transition_data,  # transition_data
                slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),  # 取buffer中指定的batch
                slice(0, ep_batch.max_seq_length),  # 指定时间片
                mark_filled=False)

            self.update(
                ep_batch.data.episode_data,  # episode_data
                slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))

            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)  # episodes_in_buffer不会超过buffer_size
            self.buffer_index = self.buffer_index % self.buffer_size  # buffer填满了就从头开始更新
            assert self.buffer_index < self.buffer_size

        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]  # 调用__getitem__，传入slices对象

        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)  # 在[0,episodes_in_buffer]取batch_size个数据，replace=F 不可重复
            return self[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer, self.buffer_size, self.scheme.keys(), self.groups.keys())
