import os
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, models

from utils import test, load_data_minerals_test, try_gpu, compute_adjustment, ClassifierFC, load_data_minerals


def main(args):
    device = try_gpu(0)  # 尝试使用 GPU
    print(args)
    print("运算设备为 {}".format(device))

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_loader_all, test_loader_list = load_data_minerals_test(data_transform=data_transform)

    if args.need_post_logit_adj:
        loader_for_num, = *load_data_minerals(batch_size=args.batch_size,
                                              max_num_workers=4,
                                              pin_memory=True,
                                              just_train=True),
    else:
        loader_for_num = None

    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"文件 '{json_path}' 不存在"

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = models.resnet50()
    feat_dim = model.fc.in_features
    if args.weights_classifier != '' and args.is_ReLU:
        model.fc = nn.ReLU()  # 原论文做法
    elif args.weights_classifier != '' and not args.is_ReLU:
        model.fc = torch.nn.Identity()  # 删除分类头
    else:
        model.fc = nn.Linear(feat_dim, args.num_classes)  # 修改最后一层

    assert os.path.exists(args.weights), "文件 {} 不存在".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    model.to(device)

    if args.weights_classifier != '':
        classifier = ClassifierFC(feat_dim=feat_dim, num_classes=args.num_classes, is_bias=args.is_bias)  # 创建分类器
        assert os.path.exists(args.weights_classifier), "文件 {} 不存在".format(args.weights_classifier)
        classifier.load_state_dict(torch.load(args.weights_classifier, map_location='cpu'))
        classifier.to(device)
    else:
        classifier = None

    if args.need_post_logit_adj:
        logit_adjustments = compute_adjustment(loader_for_num, args.tro, device)
    else:
        logit_adjustments = None

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--need_post_logit_adj', help='是否开启事后 logits 调整', type=bool, default=False)
    parser.add_argument('--tro', type=float, help='事后 logits 调整参数 tro', default=2.4)
    parser.add_argument('--is_bias', help='分类头是否带偏置', type=bool, default=True)
    parser.add_argument('--is_ReLU', help='是否换 relu', type=bool, default=False)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/ResNet50/018/weights'
                        #         '/model-end.pth',
                        default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/logit_adj/loss/006/61-99'
                                '/解耦/2/weights/model-7.pth',
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/028/61-99 37/lal/解耦/weights/'
                        #         'model-9.pth',
                        # default='./weights/model-3.pth',
                        help='initial weights path')

    parser.add_argument('--weights_classifier', type=str,
                        # default='',
                        default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/logit_adj/loss/006/61-99'
                                '/解耦/2/weights/classifier-7.pth',
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/028/61-99 37/lal/解耦/weights/classifier'
                        #         '-9.pth',
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/028/61-99 6/la/解耦/weights/'
                        #         'classifier-3.pth',
                        # default='./weights/classifier-3.pth',
                        help='initial weights path')

    opt = parser.parse_args()

    main(opt)
