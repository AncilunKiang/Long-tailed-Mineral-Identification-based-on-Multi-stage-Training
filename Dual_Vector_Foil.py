import os
import sys
import json
import pickle
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms, models

from utils import test, load_data_minerals_test, try_gpu, compute_adjustment, ClassifierFC

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

    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)

    assert os.path.exists(args.weights), "文件 {} 不存在".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    model.to(device)

    if args.need_post_logit_adj:
        logit_adjustments = compute_adjustment(test_loader_all, args.tro, device)
    else:
        logit_adjustments = None

    if args.is_test:

        if args.weights_classifier != '':
            classifier = ClassifierFC(feat_dim=2, num_classes=args.num_classes, is_bias=args.is_bias)  # 创建分类器
            assert os.path.exists(args.weights_classifier), "文件 {} 不存在".format(args.weights_classifier)
            classifier.load_state_dict(torch.load(args.weights_classifier, map_location='cpu'))
            classifier.to(device)
        else:
            classifier = None

        # 总体测试
        acc = test(model=model,
                   data_loader=test_loader_all,
                   device=device,
                   logit_adjustments=logit_adjustments,
                   classifier=classifier)

        print('测试集总精度精度: %.3f' % acc)

        # 分类别测试
        acc_list = []
        for i, loader in enumerate(test_loader_list):
            acc = test(model=model,
                       data_loader=loader,
                       device=device,
                       class_name=class_indict[str(i)],
                       logit_adjustments=logit_adjustments,
                       classifier=classifier)
            acc_list.append(acc)
        print(f"单类别精度分别为：{acc_list}")

        print('测试完成')

    if args.is_two_dimensional_transformation:
        print(f"开始降维打击")
        coordinate_list = []
        for i, loader in enumerate(test_loader_list):
            coordinate_list_single = []

            model.eval()  # 开启验证模式

            with torch.no_grad():
                pbar = tqdm(total=len(loader), file=sys.stdout)  # 设置进度条
                for batch, data in enumerate(loader):
                    inputs, labels = data  # 获取输入和标签
                    inputs, labels = inputs.to(device), labels.to(device)  # 数据转移至指定设备运算

                    coordinate = model(inputs)

                    coordinate_list_single.append(coordinate[0].tolist())

                    pbar.set_description("[Transforming {:>13s} into Two-Dimensions]".format(class_indict[str(i)]))
                    pbar.update()  # 更新进度条

                pbar.close()  # 关闭进度条
            coordinate_list.append(coordinate_list_single)

        print(f"降维打击完成")

        print('正在保存坐标点们')
        with open('coordinate_list.pkl', 'wb') as file:
            pickle.dump(coordinate_list, file)
        print('已保存至 ./coordinate_list.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--need_post_logit_adj', help='是否开启事后 logits 调整', type=bool, default=False)
    parser.add_argument('--tro', type=float, help='事后 logits 调整参数 tro', default=2.4)
    parser.add_argument('--is_bias', help='分类头是否带偏置', type=bool, default=True)
    parser.add_argument('--is_test', help='是否测试', type=bool, default=True)
    parser.add_argument('--is_two_dimensional_transformation', help='是否二维化', type=bool, default=False)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/ResNet50/012/weights'
                                '/model-60.pth',
                        help='initial weights path')

    parser.add_argument('--weights_classifier', type=str,
                        # default='',
                        default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/ResNet50/012/weights'
                                '/classifier-60.pth',
                        help='initial weights path')

    opt = parser.parse_args()

    main(opt)
