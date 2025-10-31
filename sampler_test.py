import os
import argparse

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils import source_import
from collections import Counter
import pandas as pd


from utils import try_gpu, load_data_minerals


def main(args):
    device = try_gpu()  # 尝试使用 GPU
    print(args)
    print("using {} device.".format(device))

    if os.path.exists("./weights") is False:  # 设置权重记录文件夹
        os.makedirs("./weights")

    data_transform = {   # 预处理
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                     transforms.RandomHorizontalFlip(),  # 随机翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_loader, validate_loader = load_data_minerals(batch_size=args.batch_size,
                                                       data_transform=data_transform,
                                                       sampler="ClassAwareSampler",
                                                       num_samples_cls=args.num_samples_cls)

    class_names = train_loader.dataset.classes  # 获取所有类别的名称
    class_counts_df = pd.DataFrame(columns=["batch", "mineral", "num"])
    # 遍历 train_loader
    for i, (images, labels) in enumerate(train_loader):
        # 计算当前批次中每个类别的样本数量
        class_counts = Counter(labels.numpy())
        # 将当前批次的类别计数添加到DataFrame中
        for class_name, count in class_counts.items():
            class_counts_df = pd.concat([class_counts_df, pd.DataFrame([[i+1, class_names[class_name], count]], columns=['batch', 'mineral', 'num'])], ignore_index=True)
        # 如果只想检查前几个批次，可以在这里添加一个 break 语句
        # if i >= 20: break
    # 如果您想查看每个批次的具体计数，可以打印整个DataFrame
    # print(class_counts_df)
    output_file_path = 'class_counts.csv'
    class_counts_df.to_csv(output_file_path)
    print(f"DataFrame saved to {output_file_path}")

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=256)
    # parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.042)
    parser.add_argument('--eta-min', type=float, default=0.00042)
    parser.add_argument('--num_samples_cls', type=int, default=4)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./resnet50-pre.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()

    main(opt)
