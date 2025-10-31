"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import random
import numpy as np
from torch.utils.data.sampler import Sampler
import pdb

##################################
## Class-aware sampling, partly implemented by frombeijingwithlove
##################################

class RandomCycleIter:
    # 返回包含类别样本索引循环迭代器的列表，每个循环迭代器对应一个类别的样本索引
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self  # 定义迭代器方法，返回自身作为迭代器
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:  # 如果索引等于数据集长度，表示已到达数据集末尾
            self.i = 0  # 重置索引为 0
            if not self.test_mode:
                random.shuffle(self.data_list)  # 非测试模式下随机打乱数据集列表
            
        return self.data_list[self.i]  # 返回当前索引对应的元素
    
def class_aware_sample_generator (cls_iter, data_iter_list, n, num_samples_cls=1):

    i = 0  # 计数已生成的样本数量
    j = 0  # 记录当前类别中已经生成的样本数量
    while i < n:  # 生成样本数量 为 n
        
#         yield next(data_iter_list[next(cls_iter)])
        
        if j >= num_samples_cls:  # j 大于等于 num_samples_cls，则重置 j 为 0，表示需要切换到下一个类别。
            j = 0
    
        if j == 0:  # j 等于 0，则表示需要生成当前类别的新样本集合
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

class ClassAwareSampler (Sampler):
    
    def __init__(self, data_source, num_samples_cls=1,):
        # num_classes = len(np.unique(data_source.labels))
        num_classes = len(np.unique(data_source.targets))  # 计算数据集中的类别数量 ImageFolder 通常没有直接的 labels,此处可使用 targets 替代
        self.class_iter = RandomCycleIter(range(num_classes))  # 创建一个循环迭代器，用于循环遍历类别索引
        cls_data_list = [list() for _ in range(num_classes)]  # 其中每个元素对应一个类别，存储该类别的样本索引
        # for i, label in enumerate(data_source.labels):
        for i, label in enumerate(data_source.targets):  # ImageFolder 通常没有直接的 labels,此处可使用 targets 替代
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]  # 循环迭代器列表，其中每个元素对应一个类别的样本索引循环迭代器
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)  # 总的样本数量，它等于每个类别样本数量的最大值乘以类别数量，以确保每个类别都有足够的样本被包含
        self.num_samples_cls = num_samples_cls  # 每个类别中每个batch要包含的样本数量
        
    def __iter__ (self):  # __iter__ 方法会被 DataLoader 调用，以生成每个epoch中的样本索引
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    def __len__ (self):
        return self.num_samples
    
def get_sampler():
    return ClassAwareSampler

##################################