import os
import sys
import argparse
import pickle
import random
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler


from utils import evaluate, load_data_minerals, CenterLoss, try_gpu,\
    get_center_weight, ClassifierFC, CombinedModel, update_env_ERM


def main(args):
    device = try_gpu()  # 尝试使用 GPU
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
                                                       pin_memory=True,
                                                       need_index_train=True,
                                                       sampler="DistributionSampler",
                                                       batch_split=args.batch_split)

    model = models.resnet50()  # 定义模型

    feat_dim = model.fc.in_features
    if args.is_ReLU:
        model.fc = nn.ReLU()  # 原论文做法
    else:
        model.fc = torch.nn.Identity()  # 删除分类头

    if args.weights_model != "":  # 预训练权重不为空则进行载入
        assert os.path.exists(args.weights_model), "file {} does not exist.".format(args.weights_model)
        model.load_state_dict(torch.load(args.weights_model, map_location='cpu'))

    if args.freeze_layers:  # 冻结权重
        print("开启冻结")
        for name, para in model.named_parameters():
            para.requires_grad_(False)

    model.to(device)

    classifier = ClassifierFC(feat_dim=feat_dim, num_classes=args.num_classes, is_bias=args.is_bias)  # 创建分类器

    if args.weights_classifier != "":  # 预训练权重不为空则进行载入
        assert os.path.exists(args.weights_classifier), "file {} does not exist.".format(args.weights_classifier)
        classifier.load_state_dict(torch.load(args.weights_classifier, map_location='cpu'))
    classifier.to(device)

    tb_writer.add_graph(CombinedModel(model, classifier), torch.zeros((1, 3, 224, 224), device=device))  # 记录模型结构

    pg_model = [p for p in model.parameters() if p.requires_grad]  # 特征提取器需要训练的参数
    pg_classifier = [p for p in classifier.parameters() if p.requires_grad]  # 分类头需要训练的参数
    optimizer = optim.SGD(pg_model + pg_classifier, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # 5e-5
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)  # 使用余弦衰减

    loss_function = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失
    loss_center = CenterLoss(num_classes=args.num_classes, feat_dim=feat_dim, device=device)  # 中心损失
    nn.init.kaiming_normal_(loss_center.centers, nonlinearity='relu')
    center_optimizer = torch.optim.SGD(loss_center.parameters(), lr=args.lr_ct)  # 中心损失优化器

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        print("同步学习率")
        for epoch in range(args.epochs_begin + 1):  # 同步学习率
            scheduler.step()

    env1_loader, env2_loader = train_loader  # 双环境
    assert len(env1_loader) == len(env2_loader)

    update_env_ERM(env2_loader)

    best_acc = 0.0
    for epoch in range(args.epochs_begin+1, args.epochs):  # 训练

        center_weight = get_center_weight(epoch, args.center_weights, args.center_milestones)

        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_loss_ce = torch.zeros(1).to(device)  # 累计主损失
        accu_loss_ct = torch.zeros(1).to(device)  # 累计中心损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

        sample_num = 0
        train_loss = train_loss_ce = train_loss_ct = 0.0  # 总损失 主损失 中心损失
        train_acc = 0.0  # 总精度 环境1精度 环境2精度
        pbar = tqdm(total=len(env1_loader), file=sys.stdout)  # 设置手动进度条
        for step, ((inputs1, labels1, indexs1), (inputs2, labels2, indexs2)) in enumerate(zip(env1_loader, env2_loader)):
            model.train()
            classifier.train()
            optimizer.zero_grad()

            inputs = torch.cat([inputs1, inputs2], dim=0).to(device)
            labels = torch.cat([labels1, labels2], dim=0).to(device)
            indexs = torch.cat([indexs1, indexs2], dim=0).to(device)

            sample_num += inputs.shape[0]

            features = model(inputs)  # 特征
            pred = classifier(features)  # logits，add_inputs 在这儿没啥用

            # 计数各种预测正确的样本数
            pred_classes = torch.max(pred, dim=1)[1]  # 获取预测值
            accu_num += torch.eq(pred_classes, labels).sum()  # 累计预测正确的样本数

            # 计算损失们
            loss_ce = loss_function(pred, labels)  # 计算主损失
            center_optimizer.zero_grad()
            loss_ct = loss_center(features, labels) * center_weight  # 计算中心损失

            # 反向传播
            loss = loss_ce + loss_ct
            loss.backward()

            optimizer.step()

            for param in loss_center.parameters():
                param.grad.data *= (1. / (center_weight + 1e-12))  # 乘以（1./α），以消除 α 对更新中心的影响

            center_optimizer.step()

            # 累计各种损失
            accu_loss += loss.detach()
            accu_loss_ce += loss_ce.detach()
            accu_loss_ct += loss_ct.detach()

            # 计算本轮次的各种平均损失
            train_loss = accu_loss.item() / (step + 1)
            train_loss_ce = accu_loss_ce.item() / (step + 1)
            train_loss_ct = accu_loss_ct.item() / (step + 1)
            # 计算本轮次的各种精度
            train_acc = accu_num.item() / sample_num
            # 打到进度条上
            pbar.set_description("[train epoch {}] loss: {:.3f}, loss_ce: {:.3f}, loss_ct: {:.3f}, "
                                 "acc: {:.3f}".format(epoch, train_loss, train_loss_ce, train_loss_ct, train_acc))
            pbar.update()

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                pbar.close()
                sys.exit(1)
        pbar.close()

        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model=model,
                                     loss_function=loss_function,
                                     classifier=classifier,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch)

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_loss_ce", train_loss_ce, epoch)
        tb_writer.add_scalar("train_loss_ct", train_loss_ct, epoch)
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

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--epochs_begin', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)  # 256
    parser.add_argument('--lr', type=float, default=0.042)  # 0.01417
    parser.add_argument('--eta-min', type=float, default=0.0)
    parser.add_argument('--pth_save_frequency', type=int, default=5)
    parser.add_argument('--batch_split', help='环境 batch 分割', type=bool, default=True)
    parser.add_argument('--max_num_workers', help='最大加载线程数', type=int, default=4)
    parser.add_argument('--center_weights', help='折衷参数 α', type=list, default=[0.0, 0.003, 0.005])
    parser.add_argument('--center_milestones', help='折衷参数跳变点', type=list, default=[0, 60, 80])
    parser.add_argument('--update_milestones', help='环境更新点', type=list, default=[60, 80])  # [60, 80]
    parser.add_argument('--sample_scale', help='采样间隔', type=float, default=4.0)
    parser.add_argument('--seed', default=3407, type=int, help='Fix the random seed for reproduction. Default is 25.')
    parser.add_argument('--lr_ct', type=float, default=0.5)  # 0.5
    parser.add_argument('--is_bias', help='分类头是否带偏置', type=bool, default=False)
    parser.add_argument('--is_ReLU', help='是否换 relu', type=bool, default=True)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights_model', type=str,
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/013/weights/model'
                        #         '-60.pth',
                        default='./model-60-013.pth',
                        help='initial weights path')
    parser.add_argument('--weights_classifier', type=str,
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/013/weights/classifier'
                        #         '-60.pth',
                        default='./classifier-60-013.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()

    main(opt)
