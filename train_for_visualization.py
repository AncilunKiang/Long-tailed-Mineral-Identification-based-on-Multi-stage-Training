import os
import sys
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler


from utils import evaluate, load_data_minerals, try_gpu, ClassifierFC, CombinedModel


def main(args):
    device = try_gpu(1)  # 尝试使用 GPU
    print(args)
    print("运算设备为 {}".format(device))

    # 固定随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists("./weights") is False:  # 设置权重记录文件夹
        os.makedirs("./weights")

    if os.path.exists("./runs/logs") is False:  # 设置 tensorboard 记录文件夹
        os.makedirs("./runs/logs")

    tb_writer = SummaryWriter(log_dir="runs/logs")  # 实例化 SummaryWriter 对象

    # 加载矿物数据集
    train_loader, validate_loader = load_data_minerals(batch_size=args.batch_size,
                                                       max_num_workers=args.max_num_workers,
                                                       pin_memory=True)

    model = models.resnet50()  # 定义模型

    feat_dim = model.fc.in_features
    if args.freeze_layers:  # 启用已训练好的特征提取器
        model.fc = torch.nn.Identity()  # 删除分类头

    if args.weights != "":  # 预训练权重不为空则进行载入
        assert os.path.exists(args.weights), "文件 {} 不存在".format(args.weights)
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    model.fc = nn.Linear(feat_dim, 2)
    nn.init.kaiming_normal_(model.fc.weight, nonlinearity='relu')  # 对 fc 层的权重进行 Kaiming 初始化

    if args.freeze_layers:  # 冻结权重
        print("开启冻结")
        trained_parameters_list = []
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)  # 除 fc 外，其他权重全部冻结
            else:
                trained_parameters_list.append(name)
        assert len(trained_parameters_list) > 0, "没有参数参与训练"
        print("仅训练：{}".format(trained_parameters_list))

    model.to(device)

    classifier = ClassifierFC(feat_dim=2, num_classes=args.num_classes, is_bias=args.is_bias)  # 创建分类器

    nn.init.kaiming_normal_(classifier.fc.weight, nonlinearity='relu')  # 对 fc 层的权重进行 Kaiming 初始化
    if args.is_bias:
        nn.init.constant_(classifier.fc.bias, 0)  # 将 fc 层的偏置初始化为0
    classifier.to(device)

    tb_writer.add_graph(CombinedModel(model, classifier), torch.zeros((1, 3, 224, 224), device=device))  # 记录模型结构

    pg_model = [p for p in model.parameters() if p.requires_grad]  # 特征提取器需要训练的参数
    pg_classifier = [p for p in classifier.parameters() if p.requires_grad]  # 分类头需要训练的参数
    optimizer = optim.SGD(pg_model + pg_classifier, lr=args.lr, momentum=0.9, weight_decay=5e-5)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
                                        pct_start=args.warmup_epochs / args.epochs,
                                        final_div_factor=args.final_div_factor)

    loss_function = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失

    best_acc = 0.0
    for epoch in range(args.epochs):  # 训练
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

        sample_num = 0
        train_loss = 0.0  # 总损失
        train_acc = 0.0  # 总精度
        pbar = tqdm(total=len(train_loader), file=sys.stdout)  # 设置手动进度条
        for step, (inputs, labels) in enumerate(train_loader):
            model.train()
            classifier.train()
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)

            sample_num += inputs.shape[0]  # 样本量计数

            features = model(inputs)  # 特征
            pred = classifier(features)  # logits，add_inputs 在这儿没啥用

            # 计数各种预测正确的样本数
            pred_classes = torch.max(pred, dim=1)[1]  # 获取预测值
            accu_num += torch.eq(pred_classes, labels).sum()  # 累计预测正确的样本数

            # 计算损失们
            loss = loss_function(pred, labels)  # 计算主损失

            # 反向传播
            loss.backward()

            optimizer.step()

            # 累计损失
            accu_loss += loss.detach()

            # 计算本轮次的各种平均损失
            train_loss = accu_loss.item() / (step + 1)
            # 计算本轮次的各种精度
            train_acc = accu_num.item() / sample_num
            # 打到进度条上
            pbar.set_description("[train epoch {}] loss: {:.3f}, acc: {:.3f}, "
                                 "lr: {}".format(epoch,
                                                 train_loss,
                                                 train_acc,
                                                 optimizer.param_groups[0]["lr"]))
            pbar.update()

            scheduler.step()

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                pbar.close()
                sys.exit(1)
        pbar.close()

        # scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model=model,
                                     loss_function=loss_function,
                                     classifier=classifier,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch)

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if epoch % args.pth_save_frequency == 0:  # 每 args.pth_save_frequency 轮 best_acc 清零 保存一次模型
            best_acc = 0.0
        if val_acc > best_acc:  # 每 args.pth_save_frequency 轮保存最佳模型
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(int(epoch / args.pth_save_frequency)))
            torch.save(classifier.state_dict(), "./weights/classifier-{}.pth".format(int(epoch / args.pth_save_frequency)))

    torch.save(model.state_dict(), "./weights/model-end.pth")
    torch.save(classifier.state_dict(), "./weights/classifier-end.pth")

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)  # 256
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--eta-min', type=float, default=0.00042)
    parser.add_argument('--pth_save_frequency', type=int, default=10)
    parser.add_argument('--max_num_workers', help='最大加载线程数', type=int, default=4)
    parser.add_argument('--seed', default=3407, type=int, help='Fix the random seed for reproduction. Default is 25')
    parser.add_argument('--is_bias', help='分类头是否带偏置', type=bool, default=True)
    parser.add_argument('--warmup_epochs', default=10, type=float, help='warmup 阶段')
    parser.add_argument('--final_div_factor', default=1e5, type=float, help='weight decay (default: 1e-4)')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        default='./model-end-bl-018.pth',
                        # default='./resnet50-011-model-9.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)

    opt = parser.parse_args()

    main(opt)
