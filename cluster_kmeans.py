import os
import sys
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from torchvision import transforms, models


from utils import load_data_minerals_test, try_gpu


def main(args):
    # 尝试使用 GPU
    device = try_gpu(0)
    print(f"运算设备为 {device}")

    # 定义数据预处理变换
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载测试数据加载器
    test_loader_all, test_loader_list = load_data_minerals_test(data_transform=data_transform, need_index=True)

    # 加载类别索引字典
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"文件 '{json_path}' 不存在"

    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    # 初始化 ResNet50 模型
    model = models.resnet50()

    if args.is_ReLU:
        model.fc = nn.ReLU()  # 原论文做法
    else:
        model.fc = torch.nn.Identity()  # 删除分类头

    if args.weights_model != "":  # 预训练权重不为空则进行载入
        assert os.path.exists(args.weights_model), "file {} does not exist.".format(args.weights_model)
        model.load_state_dict(torch.load(args.weights_model, map_location='cpu'))

    # 将模型移动到指定设备并设置为评估模式
    model.to(device)
    model.eval()

    # 存储聚类结果的字典
    clustering_results = {}

    # 确定要进行聚类的类别
    if args.selected_classes is None or len(args.selected_classes) == 0:
        selected_loaders = test_loader_list
        selected_class_names = list(class_indict.values())
    else:
        selected_class_names = args.selected_classes
        selected_loaders = [test_loader_list[i] for i in range(len(test_loader_list)) if class_indict[str(i)] in selected_class_names]

    # os.environ["OMP_NUM_THREADS"] = '2'

    # 对选定的类别进行特征提取和聚类
    for i, loader in enumerate(selected_loaders):
        class_name = selected_class_names[i]
        print(f"正在处理类别: {class_name}")

        features = {}
        labels = []
        with torch.no_grad():
            pbar = tqdm(total=len(loader), desc=f"Extracting Features for {class_name}", file=sys.stdout)
            for inputs, label, index in loader:
                inputs = inputs.to(device)
                feature = model(inputs)
                path = loader.dataset.dataset.samples[index][0]
                path = 'test\\' + path.split('test\\')[-1]
                features[path] = feature.cpu().numpy()
                labels.append(label.item())
                pbar.update()
            pbar.close()

        # features = np.concatenate(features)
        features_con = np.concatenate(list(features.values()))
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto').fit(features_con)
        clustering_results[class_name] = list(zip(kmeans.labels_.tolist(), list(features.keys()), labels))

    # 保存聚类结果到 JSON 文件
    output_path = 'clustering_results_12.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clustering_results, f, indent=4)

    print(f"聚类结果已保存到 {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', type=int, default=12, help='每个类别的簇数量')
    parser.add_argument('--selected_classes', default=[], type=list, help='要聚类的具体类别名称列表（留空则对所有类别聚类）')
    parser.add_argument('--is_ReLU', type=bool, default=False, help='是否换 ReLU')

    parser.add_argument('--weights_model', type=str,
                        default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/ResNet50/018/weights/model'
                                '-end.pth',
                        help='initial weights path')

    opt = parser.parse_args()

    main(opt)






