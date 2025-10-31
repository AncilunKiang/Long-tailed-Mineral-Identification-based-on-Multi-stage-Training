import os
import argparse
import pickle

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils import source_import
from collections import Counter
import pandas as pd


from utils import try_gpu, load_data_minerals, get_priority


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
                                                       need_index=True,
                                                       sampler="ClassPrioritySampler")

    training_data_num = len(train_loader.dataset)
    epoch_steps = int(training_data_num / args.batch_size)

    for epoch in range(100):
        for step, (images, labels, indexes) in enumerate(train_loader):
            if step == epoch_steps:
                break

            # 前向传播

            ptype = train_loader.sampler.ptype
            logits = torch.randn(args.batch_size, args.num_classes)  # 测试 logits
            ws = get_priority(ptype, logits.detach(), labels)
            inlist = [indexes.cpu().numpy(), ws, labels.cpu().numpy()]
            train_loader.sampler.update_weights(*inlist)

        weight = {key: value.tolist() for key, value in train_loader.sampler.get_weights().items()}
        with open('dict.pkl', 'wb') as file:  # 写入字典到文件
            pickle.dump(weight, file)

        train_loader.sampler.reset_weights(epoch)

    output_file_path = 'class_counts.csv'
    # class_counts_df.to_csv(output_file_path)
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
