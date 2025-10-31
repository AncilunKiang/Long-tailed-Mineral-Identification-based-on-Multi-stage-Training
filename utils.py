import os
import sys
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import Subset, DataLoader

import numpy as np
from tqdm import tqdm
import importlib


def try_gpu(i=0):
    """
    尝试 使用 GPU 如果存在，则返回 GPU(i)，否则返回 CPU()
    :param i: GPU 号
    :return: 设备
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ImageFolderWithIndex(datasets.ImageFolder):
    """
    多返回一个下标版本的 ImageFolder
    """
    def __init__(self, root, transform=None, augment_transform=None, augment_classes_list=None):
        assert (augment_transform and augment_classes_list) or (
                    not augment_transform and not augment_classes_list), "增强参数不足"
        super(ImageFolderWithIndex, self).__init__(root)

        self.base_transform = transform
        self.augment_transform = augment_transform
        self.augment_classes_list = augment_classes_list if augment_classes_list is not None else []

    def __getitem__(self, index):
        sample, target = super(ImageFolderWithIndex, self).__getitem__(index)  # 首先调用原始的 __getitem__ 方法

        if self.augment_classes_list and target in self.augment_classes_list:
            sample = self.augment_transform(sample)
        else:
            sample = self.base_transform(sample)

        return sample, target, index  # 多返回一个下标


class DistributionSampler(Sampler):
    """
    来自 IFL 的权重采样器
    """
    def __init__(self, dataset):
        self.num_samples = len(dataset)
        self.indexes = torch.arange(self.num_samples)
        self.weight = torch.zeros_like(self.indexes).fill_(1.0).float()  # init weight

    def __iter__(self):
        self.prob = self.weight / self.weight.sum()

        indices = torch.multinomial(self.prob, self.num_samples, replacement=True).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_parameter(self, weight):
        self.weight = weight.float()

    def get_weight(self):
        return self.weight


def load_data_minerals_test(data_transform, batch_size=1, max_num_workers=8, pin_memory=False, need_index=False):
    """
    加载矿物数据集的测试集
    :param data_transform: 预处理策略
    :param batch_size: 批量大小
    :param max_num_workers:  载入数据时的最大线程数，默认为 8
    :param pin_memory:  载入数据时是否拷贝数据到 CUDA Pinned Memory，有可能会提速，默认关闭
    :param need_index: 是否需要下标
    :return: 返回测试集和分类别的测试集列表
    """
    test_path = os.path.join(os.path.join(os.getcwd(), "../../"), "data", "minerals", "test")  # 矿物数据集位置 原 "../"
    assert os.path.exists(test_path), "{} 路径不存在".format(test_path)

    if need_index:
        test_dataset = ImageFolderWithIndex(root=test_path,  # 读取测试集文件夹，载入测试集 datasets，需要载入下标
                                            transform=data_transform)
    else:
        test_dataset = datasets.ImageFolder(root=test_path,  # 读取测试集文件夹，载入测试集 datasets，无需下标
                                            transform=data_transform)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, max_num_workers])  # 确定载入线程数 num_workers

    print('使用 {} 个线程加载数据，测试集图像数量：{}'.format(nw, len(test_dataset)))

    index_list = []  # 存储各个类别的起始/结束下标
    begin = end = 0
    # 逐个获取文件夹长度，换算为起始下标与结束下标
    for length in [len(os.listdir(os.path.join(test_path, folder_name))) for folder_name in test_dataset.classes]:
        end += length
        index_list.append([begin, end])
        begin = end

    return (DataLoader(test_dataset, batch_size=batch_size, num_workers=nw, pin_memory=pin_memory),
            [DataLoader(Subset(test_dataset, range(index[0], index[1])), batch_size=batch_size, num_workers=nw, pin_memory=pin_memory)
             for index in index_list])


def load_data_minerals(image_path_train="../../data/minerals/train", image_path_val="../../data/minerals/val",  # 原 ../
                       data_transform_train=transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                                                transforms.RandomHorizontalFlip(),  # 随机翻转
                                                                transforms.ToTensor(),
                                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])]),
                       data_transform_val=transforms.Compose([transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                                   [0.229, 0.224, 0.225])]),
                       augment_transform=None, augment_classes_list=None,
                       batch_size=32, max_num_workers=8, pin_memory=False,
                       need_index_train=False, need_index_val=False,
                       sampler=None, num_samples_cls=1,
                       batch_split=True, just_env0=False,
                       just_train=False, just_val=False):
    """
    加载矿物数据集的训练集和验证集
    :param image_path_train: 训练集图片路径
    :param image_path_val: 验证集图片路径
    :param data_transform_train: 训练集数据预处理策略
    :param data_transform_val: 验证集数据预处理策略
    :param augment_transform: 选择性增强策略
    :param augment_classes_list: 应用选择性增强策略的类别
    :param batch_size: 批量大小
    :param max_num_workers: 载入数据时的最大线程数，默认为 8
    :param pin_memory: 载入数据时是否拷贝数据到 CUDA Pinned Memory，有可能会提速，默认关闭
    :param need_index_train: 训练集是否需要下标
    :param need_index_val: 验证集是否需要下标
    :param sampler: 采样器，为 None 时则为默认采样器，可选参数：None、"ClassAwareSampler"、"ClassPrioritySampler"、"DistributionSampler"
    :param num_samples_cls: 采样单位，默认为 1
    :param batch_split: IFL 双环境下是否平分 batch_size
    :param just_env0: 是否仅加载 IFL 单环境
    :param just_train: 是否仅加载训练集
    :param just_val: 是否仅加载验证集
    :return: 返回训练集和验证集（IFL 的训练集为双环境，仅单环境的 IFL 则只返回单个训练集）
    """
    assert os.path.exists(image_path_train), "{} 路径不存在".format(image_path_val)
    assert not just_train or not just_val, "无数据加载行为"

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, max_num_workers])  # 确定载入线程数 num_workers
    print("使用 {} 个线程加载数据  ".format(nw), end='')

    data_loaders = ()  # 最终返回 data_loader 元组

    if not just_val:  # 需要加载训练集
        if need_index_train:  # 需要下标
            train_dataset = ImageFolderWithIndex(root=image_path_train,  # 读取训练集文件夹，载入训练集 datasets
                                                 transform=data_transform_train,
                                                 augment_transform=augment_transform,
                                                 augment_classes_list=augment_classes_list)
        else:  # 默认无需下标
            train_dataset = datasets.ImageFolder(root=image_path_train,  # 读取训练集文件夹，载入训练集 datasets
                                                 transform=data_transform_train)

        mineral_dict = train_dataset.class_to_idx  # 获取矿物类别字典 类名：类序号
        class_dict = dict((val, key) for key, val in mineral_dict.items())  # 反置为 类序号：类名
        json_str = json.dumps(class_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)  # 写入 json 文件

        assert sampler in [None, "ClassAwareSampler", "ClassPrioritySampler", "DistributionSampler"], "采样器错误"
        if sampler == "ClassAwareSampler":
            assert not just_env0, "无效单环境参数"
            train_shuffle = False  # 类别均衡采样器不可 shuffle
            sampler = source_import("./ClassAwareSampler.py").get_sampler()(train_dataset,
                                                                            num_samples_cls=num_samples_cls)  # 载入采样器
            print("采样器为 ClassAwareSampler  ", end='')
            data_loaders += (DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=nw,
                                        pin_memory=pin_memory, sampler=sampler),)
        elif sampler == "ClassPrioritySampler":
            assert not just_env0, "无效单环境参数"
            assert need_index_train, "未获取数据下标"
            train_shuffle = False  # 渐进均衡采样器不可 shuffle
            sampler = source_import("./ClassPrioritySampler.py").get_sampler()(train_dataset,
                                                                               balance_scale=1.0, fixed_scale=1.0,
                                                                               lam=None, epochs=100, cycle=0,
                                                                               nroot=None, manual_only=True,
                                                                               rescale=False, root_decay=None,
                                                                               decay_gap=30, ptype='score',
                                                                               pri_mode='train', momentum=0.,
                                                                               alpha=1.0)
            print("采样器为 ClassPrioritySampler  ", end='')
            data_loaders += (DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=nw,
                                        pin_memory=pin_memory, sampler=sampler),)
        elif sampler == "DistributionSampler":
            assert need_index_train, "未获取数据下标"
            train_shuffle = False  # 权重采样器不可 shuffle
            sampler = DistributionSampler  # 切不可仅使用同一个采样器对象
            print("采样器为 DistributionSampler  ", end='')
            if just_env0:  # 仅使用单环境
                data_loaders += (DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=nw,
                                            pin_memory=pin_memory, sampler=sampler(train_dataset)),)
            else:
                if batch_split:  # 双环境分割 batch_size
                    batch_size_train = batch_size // 2
                else:
                    batch_size_train = batch_size

                data_loaders += (
                    (DataLoader(train_dataset, batch_size=batch_size_train, shuffle=train_shuffle, num_workers=nw,
                                pin_memory=pin_memory, sampler=sampler(train_dataset)),  # 环境 1
                     DataLoader(train_dataset, batch_size=batch_size_train, shuffle=train_shuffle, num_workers=nw,
                                pin_memory=pin_memory, sampler=sampler(train_dataset))  # 环境 2
                     ),)
        else:
            assert not just_env0, "无效单环境参数"
            train_shuffle = True
            sampler = None  # 默认使用默认采样器
            print("使用默认采样器  ", end='')
            data_loaders += (DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=nw,
                                        pin_memory=pin_memory, sampler=sampler),)
        print("训练集图像数量：{}  ".format(len(train_dataset)), end='')

    if not just_train:  # 需要加载验证集
        if need_index_val:
            validate_dataset = ImageFolderWithIndex(root=image_path_val,  # 读取验证集文件夹，载入验证集 datasets
                                                    transform=data_transform_val)
        else:
            validate_dataset = datasets.ImageFolder(root=image_path_val,  # 读取验证集文件夹，载入验证集 datasets
                                                    transform=data_transform_val)

        data_loaders += (DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw,
                                    pin_memory=pin_memory),)

        print("验证集图像数量：{}  ".format(len(validate_dataset)), end='')

    print()
    return data_loaders


def shuffle_batch(x, y):
    """对 x，y 进行 shuffle 操作"""
    index = torch.randperm(x.size(0))
    x = x[index]
    y = y[index]
    return x, y


def logits2score(logits, labels):
    """CPS 功能函数，样本 logits 转分数"""
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy()
    return score


def get_priority(logits, labels):
    """CPS 功能函数，获取优先级"""
    ws = 1 - logits2score(logits, labels)
    return ws


def compute_adjustment(train_loader, tro, device):
    """
    compute the base probabilities
    计算 logit 调整值
    """
    label_freq = {}  # 初始化标签频率字典
    pbar = tqdm(total=len(train_loader), file=sys.stdout)  # 设置进度条
    pbar.set_description("[正在计算 logits 调整值]")
    for i, data in enumerate(train_loader):  # 遍历训练数据加载器
        if len(data) == 3:  # 如果数据带下标
            inputs, target, indexes = data  # 获取输入和标签
        else:
            inputs, target = data  # 获取输入和标签
        target = target.to(device)  # 标签转移至指定设备运算
        for j in target:  # 遍历每个标签
            key = int(j.item())  # 获取标签的整数值
            label_freq[key] = label_freq.get(key, 0) + 1  # 更新标签的频率
        pbar.update()  # 更新进度条

    label_freq = dict(sorted(label_freq.items()))  # 对标签频率进行排序
    label_freq_array = np.array(list(label_freq.values()))  # 将标签频率转换为数组
    label_freq_array = label_freq_array / label_freq_array.sum()  # 归一化标签频率
    adjustments = np.log(label_freq_array ** tro + 1e-12)  # 计算调整值
    adjustments = torch.from_numpy(adjustments)  # 将调整值转换为 PyTorch 张量
    adjustments = adjustments.to(device)  # 将调整值移动到指定设备

    pbar.set_description("[logits 调整值计算完成]")
    pbar.close()  # 关闭进度条

    return adjustments


def train_one_epoch(model, loss_function, optimizer, data_loader, device, epoch,
                    end_batch=None, do_shuffle=False,
                    logit_adjustments=None, weight_decay=None,
                    classifier=None,
                    scheduler=None):
    """
    每个 epoch 的训练过程
    :param model: 模型
    :param loss_function: 损失函数
    :param optimizer: 优化器
    :param data_loader: 训练集
    :param device: 指定设备
    :param epoch: 当前轮次
    :param end_batch: 中止 batch
    :param do_shuffle: 是否需要额外做 shuffle
                       CAS 采样器无随机性，故需要额外 shuffle，但是对于本身就有随机性的 CPS 采样器来说是无需额外 shuffle 的
    :param logit_adjustments: logits 调整参数
    :param weight_decay: logits 调整参数
    :param classifier: IFL 模型的特征提取器部分和分类器部分需要分别传入
    :param scheduler: 传入学习率调度器则代表需要每步更新
    :return: 返回本轮训练的平均损失和正确率
    """
    acc_loss = torch.zeros(1).to(device)  # 累计损失
    acc_correct = torch.zeros(1).to(device)  # 累计预测正确的样本数
    acc_sample = 0  # 累计样本量

    model.train()  # 开启训练模式

    if end_batch:
        pbar = tqdm(total=end_batch, file=sys.stdout)  # 设置进度条
    else:
        pbar = tqdm(total=len(data_loader), file=sys.stdout)  # 设置进度条
    for batch, data in enumerate(data_loader):
        if end_batch and batch == end_batch:  # 如果需要则提前中止本轮训练
            break

        if len(data) == 3:  # 如果数据带下标
            inputs, labels, indexes = data  # 获取输入和标签
            inputs, labels, indexes = inputs.to(device), labels.to(device), indexes.to(device)  # 数据转移至指定设备运算
        else:
            inputs, labels = data  # 获取输入和标签
            if do_shuffle:  # 如果需要，则进行额外的 shuffle
                inputs, labels = shuffle_batch(inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)  # 数据转移至指定设备运算
            indexes = None

        acc_sample += inputs.shape[0]  # 累计样本量

        if classifier is not None:  # IFL 前向传播需要劈两半
            features = model(inputs)
            logits = classifier(features)
        else:
            logits = model(inputs)  # 前向传播

        labels_predicted = torch.max(logits, dim=1)[1]  # 获取预测标签
        acc_correct += torch.eq(labels_predicted, labels).sum()  # 累计预测正确的样本数

        if logit_adjustments is not None:  # 如果设置了 logits 调整损失则进行调整
            logits += logit_adjustments

        loss = loss_function(logits, labels)  # 计算损失

        # if logit_adjustments is not None:  # 如果设置了 logits 调整损失则进行调整
        #     loss_r = 0
        #     for parameter in model.parameters():  # 对每个模型参数
        #         loss_r += torch.sum(parameter ** 2)  # 计算L2正则化损失
        #     loss += weight_decay * loss_r  # 添加正则化损失
        # loss_r = 0
        # for parameter in model.parameters():  # 对每个模型参数
        #     loss_r += torch.sum(parameter ** 2)  # 计算L2正则化损失
        # for parameter in classifier.parameters():  # 对每个模型参数
        #     loss_r += torch.sum(parameter ** 2)  # 计算L2正则化损失
        # loss = loss + weight_decay * loss_r  # 添加正则化损失

        loss.backward()  # 反向传播
        acc_loss += loss.detach()  # 累计损失值

        if not torch.isfinite(loss):
            print(f'epoch{epoch}-batch{batch} 发生梯度爆炸，即将退出训练')
            sys.exit(1)

        optimizer.step()  # 进行梯度下降
        optimizer.zero_grad()  # 梯度清零

        if scheduler:
            pbar.set_description("[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {}".format(epoch,
                                                                                             acc_loss.item()/(batch + 1),
                                                                                             acc_correct.item()/acc_sample,
                                                                                             optimizer.param_groups[0]["lr"]))
            scheduler.step()
        else:
            pbar.set_description("[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                     acc_loss.item() / (batch + 1),
                                                                                     acc_correct.item() / acc_sample))

        pbar.update()  # 更新进度条

        # 如果 CPS 需要，则更新采样器权重
        if indexes is not None and hasattr(data_loader.sampler, 'update_weights'):
            ws = get_priority(logits.detach(), labels.to(device))
            inlist = [indexes.cpu().numpy(), ws, labels.cpu().numpy()]
            data_loader.sampler.update_weights(*inlist)

    pbar.close()  # 关闭进度条

    return acc_loss.item() / (batch + 1), acc_correct.item() / acc_sample  # 返回平均损失值和正确率


def evaluate(model, loss_function, data_loader, device, epoch,
             need_val_info=False,
             logit_adjustments=None,
             classifier=None):
    """
    每轮训练后的跑一下验证集
    :param model: 模型
    :param loss_function: 损失函数
    :param data_loader: 验证集
    :param device: 指定设备
    :param epoch: 当前轮次
    :param need_val_info: CPS 需要返回 total_logits/labels
    :param logit_adjustments: logits 调整参数
    :param classifier: IFL 模型的特征提取器部分和分类器部分需要分别传入
    :return: 返回本轮验证的平均损失和正确率
    """
    acc_loss = torch.zeros(1).to(device)  # 累计损失
    acc_correct = torch.zeros(1).to(device)  # 累计预测正确的样本数
    acc_sample = 0  # 累计样本量

    if need_val_info:  # 如果需要，则初始化存储张量
        json_path = './class_indices.json'
        assert os.path.exists(json_path), f"文件 '{json_path}' 不存在"
        total_logits = torch.empty((0, len(json.load(open(json_path, "r"))))).to(device)
        total_labels = torch.empty(0, dtype=torch.long).to(device)

    model.eval()  # 开启验证模式

    with torch.no_grad():
        pbar = tqdm(total=len(data_loader), file=sys.stdout)  # 设置进度条
        for batch, data in enumerate(data_loader):
            inputs, labels = data  # 获取输入和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 数据转移至指定设备运算

            acc_sample += inputs.shape[0]  # 累计样本量
            if classifier is not None:  # IFL 前向传播需要劈两半
                features = model(inputs)
                logits = classifier(features)
            else:
                logits = model(inputs)  # 前向传播
            labels_predicted = torch.max(logits, dim=1)[1]  # 获取预测标签
            acc_correct += torch.eq(labels_predicted, labels).sum()  # 累计预测正确的样本数

            if logit_adjustments is not None:  # 如果设置了 logits 调整损失则进行调整
                logits += logit_adjustments

            loss = loss_function(logits, labels)  # 计算损失
            acc_loss += loss.detach()  # 累计损失值

            pbar.set_description("[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                     acc_loss.item() / (batch + 1),
                                                                                     acc_correct.item() / acc_sample))
            pbar.update()  # 更新进度条

            if need_val_info:  # 如果需要则记录 logits 和 labels
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, labels))

        pbar.close()  # 关闭进度条

    if need_val_info:
        return acc_loss.item() / (batch + 1), acc_correct.item() / acc_sample, total_logits, total_labels

    return acc_loss.item() / (batch + 1), acc_correct.item() / acc_sample  # 返回平均损失值和正确率


def test(model, data_loader, device, class_name=None,
         logit_adjustments=None,
         classifier=None):
    """
    跑测试集
    :param model: 模型
    :param data_loader: 测试集
    :param device: 指定设备
    :param class_name: 当前类名，默认为空，若不为空则为单类别测试
    :param logit_adjustments: 事后 logit 调整参数
    :param classifier: IFL 模型的特征提取器部分和分类器部分需要分别传入
    :return: 返回正确率
    """
    acc_correct = torch.zeros(1).to(device)  # 累计预测正确的样本数
    acc_sample = 0  # 累计样本量

    model.eval()  # 开启验证模式

    with torch.no_grad():
        pbar = tqdm(total=len(data_loader), file=sys.stdout)  # 设置进度条
        for batch, data in enumerate(data_loader):
            inputs, labels = data  # 获取输入和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 数据转移至指定设备运算

            acc_sample += inputs.shape[0]  # 累计样本量
            if classifier is not None:  # IFL 前向传播需要劈两半
                features = model(inputs)
                logits = classifier(features)
            else:
                logits = model(inputs)  # 前向传播

            if logit_adjustments is not None:  # 如果启用事后 logits 调整
                logits -= logit_adjustments

            labels_predicted = torch.max(logits, dim=1)[1]  # 获取预测标签
            acc_correct += torch.eq(labels_predicted, labels).sum()  # 累计预测正确的样本数
            if class_name:
                pbar.set_description("[testing {:>13s}] acc: {:.3f}".format(class_name, acc_correct.item() / acc_sample))
            else:
                pbar.set_description("[testing all] acc: {:.3f}".format(acc_correct.item() / acc_sample))
            pbar.update()  # 更新进度条

        pbar.close()  # 关闭进度条

    return acc_correct.item() / acc_sample


class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, device=torch.device('cuda:0')):
        super(CenterLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).to(device))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x - center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


class ClassifierFC(nn.Module):
    def __init__(self, feat_dim, num_classes=1000, is_bias=False):
        super(ClassifierFC, self).__init__()

        self.fc = nn.Linear(feat_dim, num_classes, bias=is_bias)

    def forward(self, x, add_inputs=None):
        y = self.fc(x)
        return y


class CombinedModel(torch.nn.Module):
    def __init__(self, model, classifier):
        super(CombinedModel, self).__init__()
        self.model = model
        self.classifier = classifier

    def forward(self, x):
        # 假设model输出的是特征向量，不需要额外处理
        features = self.model(x)
        return self.classifier(features)


def get_center_weight(epoch, center_weights, center_milestones):
    center_weight = center_weights[0]
    for i, ms in enumerate(center_milestones):
        if epoch >= ms:
            center_weight = center_weights[i]
    return center_weight


def generate_intra_weight(cat_socres, total_image, tg_scale=4.0):
    # normalize
    intra_weight = torch.zeros(total_image).fill_(0.0)  # 初始化一个全零的权重向量，长度等于总图像数
    for cat, cat_items in cat_socres.items():  # 遍历每个类别及其对应的样本评分
        cat_size = len(cat_items)  # 获取当前类别的样本数量
        if cat_size < 5:  # 如果样本数量小于5
            for ind in list(cat_items.keys()):  # 为每个样本赋予相同的权重（1/类别样本数量）
                intra_weight[ind] = 1.0 / max(cat_size, 1.0)
            continue  # 跳过当前类别的其他处理
        cat_inds = list(cat_items.keys())  # 获取当前类别的样本索引
        cat_scores = torch.FloatTensor([cat_items[ind] for ind in cat_inds])  # 转换为浮点张量
        cat_scores = cat_scores - cat_scores.min()  # 减去最小值，使最小值为0
        cat_scores = cat_scores / (cat_scores.max() + 1e-9)  # 标准化，使最大值为1

        # use Pareto principle to determine the scale parameter   # 使用帕累托原则（80-20法则）来确定一个比例因子
        cat_scores = (1.0 - cat_scores).abs() + 1e-5  # 取绝对值并加上一个小数，防止除以0
        head_mean = torch.topk(cat_scores, k=int(cat_size * 0.8), largest=False)[0].mean().item()  # 计算头部20%的平均值，此处头部指类内判对概率高的样本
        tail_mean = torch.topk(cat_scores, k=int(cat_size * 0.2), largest=True)[0].mean().item()  # 计算尾部80%的平均值，此处尾部指类内判错概率高的样本
        scale = tail_mean / head_mean + 1e-5  # 计算比例因子
        exp_scale = torch.FloatTensor([tg_scale]).log() / torch.FloatTensor([scale]).log()  # 计算指数比例因子
        exp_scale = exp_scale.clamp(min=1, max=10)  # 限制比例因子的范围
        cat_scores = cat_scores ** exp_scale  # 调整分数
        cat_scores = cat_scores + 1e-12  # 防止分数为0
        cat_scores = cat_scores / cat_scores.sum()  # 正规化，使分数总和为1，这里使每个类别权重和为1，确保了类别上的平衡
        for ind, score in zip(cat_inds, cat_scores.tolist()):  # 为每个样本设置计算出的权重
            intra_weight[ind] = score
    return intra_weight


def update_env_by_score(all_ind_self, all_lab_self, all_prb_self, env1_loader, env2_loader, total_image, sample_scale):
    # seperate environments by inter-score + intra-score
    all_ind, all_lab, all_prb = all_ind_self.tolist(), all_lab_self.tolist(), all_prb_self.tolist()  # 将索引、标签和概率转换为列表
    all_cat = list(set(all_lab))  # 获取所有 unique 的类别标签
    all_cat.sort()  # 排序类别标签
    cat_socres = {cat: {} for cat in all_cat}  # 初始化一个字典，用于存储每个类别的样本评分
    all_scores = {}  # 初始化一个字典，用于存储所有样本的评分
    for ind, lab, prb in zip(all_ind, all_lab, all_prb):  # 遍历索引、标签和概率
        cat_socres[lab][ind] = prb  # 将样本的评分存储到对应类别的字典中
        all_scores[ind] = prb  # 存储所有样本的评分

    # baseline distribution  # 初始化两个环境的权重向量，长度等于总图像数，初始值为 1
    env1_score = torch.zeros(total_image).fill_(1.0)
    env2_score = torch.zeros(total_image).fill_(1.0)
    # inverse distribution  # 初始化两个环境的权重向量，长度等于总图像数，初始值为1
    intra_weight = generate_intra_weight(cat_socres, total_image, tg_scale=sample_scale)
    env2_score = env2_score * intra_weight  # 将第二个环境的权重向量与内部权重相乘，提高预测概率较低的样本的权重

    env1_loader.sampler.set_parameter(env1_score)
    env2_loader.sampler.set_parameter(env2_score)


def update_env_ERM(env_loader=None):
    '''
    尝试不适用不变风险最小化，回归经验风险最小化
    即，类别平衡采样
    '''
    print("开始环境更新流程")
    if not os.path.exists('./intra_weight.pkl'):
        num_classes = len(env_loader.dataset.classes)  # 类别数
        intra_weight = torch.zeros(len(env_loader.dataset.samples)).fill_(0.0)  # 用于记录权重，初始化一个全零的权重向量，长度等于样本总数
        num_samples = torch.zeros(num_classes).fill_(0.0)  # 用于记录各类别样本量，初始化一个全零的权重向量，长度等于总图像数
        prob_classes = torch.zeros(num_classes).fill_(1.0)  # 用于记录各类别权重，初始化一个全一的权重向量，长度等于总图像数
        pbar = tqdm(total=len(env_loader.dataset), file=sys.stdout)  # 设置手动进度条
        pbar.set_description("[正在计数]")
        for _, label, index in env_loader.dataset:
            num_samples[label] += 1.0  # 计算各类别样本量
            pbar.update()
        pbar.set_description("[计数完成]")
        pbar.close()

        prob_classes /= num_samples  # 计算各类别权重

        pbar = tqdm(total=len(env_loader.dataset), file=sys.stdout)  # 设置手动进度条
        pbar.set_description("[正在加权]")
        for _, label, index in env_loader.dataset:
            intra_weight[index] = prob_classes[label]  # 给各个样本设置权重
            pbar.update()
        pbar.set_description("[加权完成]")
        pbar.close()

        with open('intra_weight.pkl', 'wb') as file:
            pickle.dump(intra_weight.tolist(), file)
        print("权重列表保存至 ./intra_weight.pkl")
    else:
        print("读取权重文件")
        with open('intra_weight.pkl', 'rb') as file:
            intra_weight = torch.Tensor(pickle.load(file))

    env_loader.sampler.set_parameter(intra_weight)  # 更新采样器权重
    print("环境更新完成")


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(loss_fc, pred, y_a, y_b, lam):
    return lam * loss_fc(pred, y_a) + (1 - lam) * loss_fc(pred, y_b)


def mixup_center_criterion(loss_center, feat, y_a, y_b, lam):
    return lam * loss_center(feat, y_a) + (1 - lam) * loss_center(feat, y_b)


def get_update_epoch(update_epoch, update_milestones):
    for update_milestone in sorted(update_milestones)[::-1]:
        if update_epoch >= update_milestone:
            update_epoch = update_milestone
            break
    return update_epoch

