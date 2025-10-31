import os
import sys
import argparse
import random
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler


from utils import evaluate, load_data_minerals, try_gpu, compute_adjustment, ClassifierFC, CombinedModel


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

    if args.is_RandAugment:
        assert not args.is_local_RandAugment, "增强选项错误"
        # 加载矿物数据集
        env0_loader, = *load_data_minerals(batch_size=args.batch_size,
                                           max_num_workers=args.max_num_workers,
                                           pin_memory=True,
                                           need_index_train=True,
                                           sampler="DistributionSampler",
                                           batch_split=args.batch_split,
                                           just_env0=True,
                                           just_train=True,
                                           data_transform_train=transforms.Compose(
                                               [transforms.RandomResizedCrop(224),  # 随机裁剪
                                                transforms.RandomHorizontalFlip(),  # 随机翻转
                                                transforms.RandAugment(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])),
    elif args.is_local_RandAugment:
        assert args.augment_classes_list != [], "空列表错误"
        env0_loader, = *load_data_minerals(batch_size=args.batch_size,
                                           max_num_workers=args.max_num_workers,
                                           pin_memory=True,
                                           need_index_train=True,
                                           sampler="DistributionSampler",
                                           batch_split=args.batch_split,
                                           just_env0=True,
                                           just_train=True,
                                           augment_transform=transforms.Compose(
                                               [transforms.RandomResizedCrop(224),  # 随机裁剪
                                                transforms.RandomHorizontalFlip(),  # 随机翻转
                                                transforms.RandAugment(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])]),
                                           augment_classes_list=args.augment_classes_list),
    else:
        # 加载矿物数据集
        env0_loader, = *load_data_minerals(batch_size=args.batch_size,
                                           max_num_workers=args.max_num_workers,
                                           pin_memory=True,
                                           need_index_train=True,
                                           sampler="DistributionSampler",
                                           batch_split=args.batch_split,
                                           just_env0=True,
                                           just_train=True),

    validate_loader, = *load_data_minerals(batch_size=args.batch_size,
                                           max_num_workers=args.max_num_workers,
                                           pin_memory=True,
                                           sampler="DistributionSampler",
                                           just_val=True),

    if args.need_logit_adj_loss:
        loader_for_num, = *load_data_minerals(batch_size=args.batch_size,
                                              max_num_workers=4,
                                              pin_memory=True,
                                              just_train=True),
    else:
        loader_for_num = None

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

    loss_function = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(env0_loader),
                                        pct_start=args.warmup_epochs / args.epochs,
                                        final_div_factor=args.final_div_factor)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        print("同步学习率")
        for epoch in range(args.epochs_begin + 1):  # 同步学习率
            for batch in range(len(env0_loader)):
                scheduler.step()

    if args.need_logit_adj_loss:
        logit_adjustments = compute_adjustment(loader_for_num, args.tro_train, device)
    else:
        logit_adjustments = None
    del loader_for_num

    best_acc = 0.0
    for epoch in range(args.epochs_begin+1, args.epochs):  # 训练
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

        sample_num = 0
        train_loss = 0.0  # 总损失
        train_acc = 0.0  # 总精度
        pbar = tqdm(total=len(env0_loader), file=sys.stdout)  # 设置手动进度条
        for step, (inputs, labels, indexs) in enumerate(env0_loader):
            model.train()
            classifier.train()
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_num += inputs.shape[0]

            features = model(inputs)  # 特征
            pred = classifier(features)  # logits，add_inputs 在这儿没啥用

            # 计数各种预测正确的样本数
            pred_classes = torch.max(pred, dim=1)[1]  # 获取预测值
            accu_num += torch.eq(pred_classes, labels).sum()  # 累计预测正确的样本数

            if args.need_logit_adj_loss:  # 如果设置了 logits 调整损失则进行调整
                pred += logit_adjustments

            # 计算损失
            loss = loss_function(pred, labels)  # 计算主损失
            if args.need_logit_adj_loss:  # 如果设置了 logits 调整损失则进行调整
                loss_r = 0
                for parameter in model.parameters():  # 对每个模型参数
                    loss_r += torch.sum(parameter ** 2)  # 计算L2正则化损失
                for parameter in classifier.parameters():  # 对每个模型参数
                    loss_r += torch.sum(parameter ** 2)  # 计算L2正则化损失
                loss = loss + args.weight_decay * loss_r  # 添加正则化损失

            # 反向传播
            loss.backward()

            optimizer.step()

            # 累计各种损失
            accu_loss += loss.detach()

            # 计算本轮次的平均损失
            train_loss = accu_loss.item() / (step + 1)
            # 计算本轮次的精度
            train_acc = accu_num.item() / sample_num
            # 打到进度条上
            pbar.set_description(
                "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {}".format(epoch, train_loss, train_acc,
                                                                            optimizer.param_groups[0]["lr"]))
            pbar.update()

            scheduler.step()

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                pbar.close()
                sys.exit(1)
        pbar.close()

        # 验证
        val_loss, val_acc = evaluate(model=model,
                                     loss_function=loss_function,
                                     classifier=classifier,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch)

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_loss_ce", train_loss, epoch)
        tb_writer.add_scalar("train_loss_ct", 0, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("train_acc_env1", train_acc, epoch)
        tb_writer.add_scalar("train_acc_env2", 0, epoch)
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

    print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--epochs_begin', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--eta-min', type=float, default=0.0)
    parser.add_argument('--pth_save_frequency', type=int, default=5)
    parser.add_argument('--batch_split', help='环境 batch 分割', type=bool, default=False)
    parser.add_argument('--max_num_workers', help='最大加载线程数', type=int, default=4)
    parser.add_argument('--seed', default=3407, type=int, help='Fix the random seed for reproduction. Default is 25.')
    parser.add_argument('--is_bias', help='分类头是否带偏置', type=bool, default=True)
    parser.add_argument('--is_ReLU', help='是否换 relu', type=bool, default=False)
    parser.add_argument('--need_logit_adj_loss', help='是否开启 logits 调整损失', type=bool, default=True)
    parser.add_argument('--tro_train', default=1.0, type=float, help='tro for logit adj train')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--warmup_epochs', default=10, type=float, help='warmup 阶段')
    parser.add_argument('--final_div_factor', default=1e5, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--is_RandAugment', help='是否启用 RandAugment', type=bool, default=False)
    parser.add_argument('--is_local_RandAugment', help='是否启用局部 RandAugment', type=bool, default=False)
    parser.add_argument('--augment_classes_list', help='局部增强类列表', type=list, default=[1, 3, 10, 11, 17, 19, 21, 27])

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights_model', type=str,
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/028/0-60/model'
                        #         '-60-028.pth',
                        default='./model-60-la-006.pth',
                        help='initial weights path')
    parser.add_argument('--weights_classifier', type=str,
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/028/0-60/classifier'
                        #         '-60-028.pth',
                        default='./classifier-60-la-006.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()

    main(opt)
