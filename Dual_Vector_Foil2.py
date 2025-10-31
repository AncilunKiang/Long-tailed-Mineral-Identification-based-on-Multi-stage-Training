import os
import sys
import json
import pickle
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms, models
import numpy as np
from sklearn.manifold import TSNE

from utils import load_data_minerals_test, try_gpu

def main(args):
    device = try_gpu()  # 尝试使用 GPU
    print(args)
    print("运算设备为 {}".format(device))

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_loader_all, test_loader_list = load_data_minerals_test(data_transform=data_transform)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"文件 '{json_path}' 不存在"

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = models.resnet50()  # 定义模型

    if args.is_ReLU:
        model.fc = nn.ReLU()  # 原论文做法
    else:
        model.fc = torch.nn.Identity()  # 删除分类头

    assert os.path.exists(args.weights_model), "file {} does not exist.".format(args.weights_model)
    model.load_state_dict(torch.load(args.weights_model, map_location='cpu'))

    if args.freeze_layers:  # 冻结权重
        print("开启冻结")
        for name, para in model.named_parameters():
            para.requires_grad_(False)

    model.to(device)

    print(f"开始降维打击")

    all_features = []
    all_labels = []
    print("获取特征向量")
    for i, loader in enumerate(test_loader_list):
        model.eval()  # 开启验证模式

        with torch.no_grad():
            pbar = tqdm(total=len(loader), file=sys.stdout)  # 设置进度条
            for batch, data in enumerate(loader):
                inputs, labels = data  # 获取输入和标签
                inputs, labels = inputs.to(device), labels.to(device)  # 数据转移至指定设备运算

                # 获取特征 (假设你的模型现在输出2048维特征)
                features = model(inputs)

                # 保存特征和标签
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_description("[get {:>13s} features]".format(class_indict[str(i)]))
                pbar.update()  # 更新进度条
            pbar.close()  # 关闭进度条
    print("特征向量获取完成")

    # 合并所有特征
    all_features = np.concatenate(all_features, axis=0)  # 形状应为 (N, 2048)
    # 第二步：全局TSNE降维
    print("正在二维化")
    tsne = TSNE(n_components=2, random_state=42)
    all_coordinates = tsne.fit_transform(all_features)  # 形状变为 (N, 2)
    print("二维化完成")

    # 第三步：按原始类别结构重组坐标
    coordinate_list = []
    start_idx = 0
    for i, loader in enumerate(test_loader_list):
        # 计算当前类别的样本数量
        num_samples = len(loader.dataset)
        # 切片获取当前类别的坐标
        class_coords = all_coordinates[start_idx:start_idx + num_samples].tolist()
        coordinate_list.append(class_coords)
        start_idx += num_samples

    print(f"降维打击完成")

    print('正在保存坐标点们')
    with open('coordinate_list.pkl', 'wb') as file:
        pickle.dump(coordinate_list, file)
    print('已保存至 ./coordinate_list.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--is_ReLU', help='是否换 relu', type=bool, default=False)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights_model', type=str,
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/ResNet50/018/weights'
                        #         '/model-end.pth',
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/028/61-99 37/weights'
                        #         '/model-end.pth',
                        default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/028/61-99 37/lal/解耦/weights'
                                '/model-9.pth',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=True)

    opt = parser.parse_args()

    main(opt)
