import os
import sys
import json
import argparse
from PIL import Image

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms, models

from utils import load_data_minerals_test, try_gpu, ClassifierFC, compute_adjustment, load_data_minerals


def main(args):
    device = try_gpu(0)  # 尝试使用 GPU
    print(args)
    print("运算设备为 {}".format(device))

    # 加载类别索引字典
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"文件 '{json_path}' 不存在"

    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    # 加载聚类结果
    clustering_results_path = 'clustering_results_9.json'
    assert os.path.exists(clustering_results_path), f"文件 '{clustering_results_path}' 不存在"

    with open(clustering_results_path, "r") as json_file:
        clustering_results = json.load(json_file)

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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

    # 计算每个类别中每个簇的识别精度
    precision_results = {}

    for i, class_name in enumerate(class_indict.values()):
        clustering_result = clustering_results[class_name]
        num_kmeans_label = max(clustering_result, key=lambda item: item[0])[0]+1
        count_right = [0] * num_kmeans_label
        count_wrong = [0] * num_kmeans_label
        pbar = tqdm(total=len(clustering_result), desc=f"Computing {class_name}", file=sys.stdout)
        for kmeans_label, path, true_label in clustering_result:
            assert true_label == i == int(path.split('_')[-1].split('.')[0]), "标签错配，请检查"

            image_path = os.path.join("../../data/minerals/", path)
            img = Image.open(image_path)

            inputs = data_transform(img)
            inputs = torch.unsqueeze(inputs, dim=0).to(device)

            model.eval()  # 开启验证模式
            with torch.no_grad():
                if classifier is not None:  # IFL 前向传播需要劈两半
                    features = model(inputs)
                    logits = classifier(features)
                else:
                    logits = model(inputs)  # 前向传播
            labels_predicted = torch.max(logits, dim=1)[1].item()  # 获取预测标签

            if labels_predicted == true_label:
                count_right[kmeans_label] += 1
            else:
                count_wrong[kmeans_label] += 1

            pbar.update()
        pbar.close()

        precision_results[class_name] = [(kmeans_label,
                                          count_right[kmeans_label]/(count_right[kmeans_label]+count_wrong[kmeans_label]),
                                          count_right[kmeans_label]+count_wrong[kmeans_label])
                                         for kmeans_label in range(num_kmeans_label)]
        precision_results[class_name].sort(key=lambda item: item[2], reverse=True)

    # 保存识别精度到 JSON 文件
    output_path_precision = 'calculate_precision_9_IFL.json'

    with open(output_path_precision, 'w', encoding='utf-8') as f:
        json.dump(precision_results, f, indent=4)

    print(f"识别精度已保存到 {output_path_precision}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--is_bias', help='分类头是否带偏置', type=bool, default=True)
    parser.add_argument('--is_ReLU', help='是否换 relu', type=bool, default=False)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/ResNet50/018/weights/model'
                        #         '-end.pth',
                        default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/032/61-99/weights/model'
                                '-end.pth',
                        help='initial weights path')

    parser.add_argument('--weights_classifier', type=str,
                        # default='',
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/ResNet50/018/weights/classifier'
                        #         '-end.pth',
                        default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/032/61-99/weights/classifier'
                                '-end.pth',
                        help='initial weights path')

    opt = parser.parse_args()

    main(opt)
