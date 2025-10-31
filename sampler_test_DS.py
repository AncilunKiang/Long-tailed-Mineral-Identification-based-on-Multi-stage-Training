import os
import sys
import argparse
import pickle
import random

import torch
from tqdm import tqdm
from torchvision import transforms


from utils import load_data_minerals, try_gpu


def main(args):
    device = try_gpu()  # 尝试使用 GPU
    print(args)
    print("运算设备为 {}".format(device))

    # 固定随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_transform = {   # 预处理
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                     transforms.RandomHorizontalFlip(),  # 随机翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 加载矿物数据集
    train_loader, validate_loader = load_data_minerals(batch_size=args.batch_size,
                                                       data_transform=data_transform,
                                                       max_num_workers=args.max_num_workers,
                                                       pin_memory=True,
                                                       need_index=True,
                                                       sampler="DistributionSampler",
                                                       batch_split=args.batch_split)

    env1_loader, env2_loader = train_loader  # 双环境
    assert len(env1_loader) == len(env2_loader)
    total_image = len(env1_loader.dataset)

    print("读取权重文件")
    with open('intra_weight.pkl', 'rb') as file:
        intra_weight = torch.Tensor(pickle.load(file))

    env2_loader.sampler.set_parameter(intra_weight)  # 更新采样器权重

    one_epoch_list = [[], []]
    pbar = tqdm(total=len(env1_loader), file=sys.stdout)  # 设置手动进度条
    pbar.set_description("[正在记录]")
    for step, ((inputs1, labels1, indexs1), (inputs2, labels2, indexs2)) in enumerate(zip(env1_loader, env2_loader)):
        one_epoch_list[0].append([labels1.tolist(), indexs1.tolist()])
        one_epoch_list[1].append([labels2.tolist(), indexs2.tolist()])
        pbar.update()

    with open('one_epoch_list.pkl', 'wb') as file:
        pickle.dump(one_epoch_list, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--batch_size', type=int, default=8)  # 256
    parser.add_argument('--batch_split', help='环境 batch 分割', type=bool, default=True)
    parser.add_argument('--max_num_workers', help='最大加载线程数', type=int, default=4)
    parser.add_argument('--sample_scale', help='采样间隔', type=float, default=4.0)
    parser.add_argument('--seed', default=3407, type=int, help='Fix the random seed for reproduction. Default is 25.')

    opt = parser.parse_args()

    main(opt)
