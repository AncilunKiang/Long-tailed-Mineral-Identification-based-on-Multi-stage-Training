import os
import argparse

import torch
import random
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler


from utils import train_one_epoch, evaluate, load_data_minerals, compute_adjustment, try_gpu


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

    if args.DistributionSampler:
        train_loader, = *load_data_minerals(batch_size=args.batch_size,
                                            max_num_workers=args.max_num_workers,
                                            pin_memory=True,
                                            need_index_train=True,
                                            sampler="DistributionSampler",
                                            just_env0=True,
                                            just_train=True),

        validate_loader, = *load_data_minerals(batch_size=args.batch_size,
                                               max_num_workers=args.max_num_workers,
                                               pin_memory=True,
                                               sampler="DistributionSampler",
                                               just_val=True),

        loader_for_num, = *load_data_minerals(batch_size=args.batch_size,
                                              max_num_workers=4,
                                              pin_memory=True,
                                              just_train=True),
    else:
        # 加载矿物数据集
        train_loader, validate_loader = load_data_minerals(batch_size=args.batch_size, max_num_workers=4)

    model = models.resnet50()  # 定义模型

    if args.weights != "":  # 预训练权重不为空则进行载入
        assert os.path.exists(args.weights), "文件 {} 不存在".format(args.weights)
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    if args.freeze_layers:  # 冻结权重
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)  # 除 fc 外，其他权重全部冻结
            else:
                print("training {}".format(name))

    model.fc = nn.Linear(model.fc.in_features, args.num_classes)  # 修改最后一层
    nn.init.kaiming_normal_(model.fc.weight, nonlinearity='relu')  # 对 fc 层的权重进行 Kaiming 初始化
    nn.init.constant_(model.fc.bias, 0)  # 将 fc 层的偏置初始化为0

    model.to(device)

    tb_writer.add_graph(model, torch.zeros((1, 3, 224, 224), device=device))  # 将模型结构写入 tensorboard

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # 优化器
    if args.use_warmup:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                            steps_per_epoch=len(train_loader),
                                            pct_start=args.warmup_epochs / args.epochs,
                                            final_div_factor=args.final_div_factor)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)  # 使用余弦衰减

    loss_function = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失

    if args.DistributionSampler:
        logit_adjustments = compute_adjustment(loader_for_num, args.tro_train, device)
    else:
        logit_adjustments = compute_adjustment(train_loader, args.tro_train, device)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # 训练
        if args.use_warmup:
            train_loss, train_acc = train_one_epoch(model=model,
                                                    loss_function=loss_function,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    logit_adjustments=logit_adjustments,
                                                    weight_decay=args.weight_decay,
                                                    scheduler=scheduler)
        else:
            train_loss, train_acc = train_one_epoch(model=model,
                                                    loss_function=loss_function,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    logit_adjustments=logit_adjustments,
                                                    weight_decay=args.weight_decay)

            scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model=model,
                                     loss_function=loss_function,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch,
                                     logit_adjustments=logit_adjustments)

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        tb_writer.add_histogram(tag="fc",
                                values=model.fc.weight,
                                global_step=epoch)

        if epoch % args.pth_save_frequency == 0:  # 每 args.pth_save_frequency 轮 best_acc 清零 保存一次模型
            best_acc = 0.0
        if val_acc > best_acc:  # 每 args.pth_save_frequency 轮保存最佳模型
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(int(epoch / args.pth_save_frequency)))

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--eta-min', type=float, default=0.00042)
    parser.add_argument('--pth_save_frequency', type=int, default=10)
    parser.add_argument('--max_num_workers', help='最大加载线程数', type=int, default=4)
    parser.add_argument('--tro_train', default=1.0, type=float, help='tro for logit adj train')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', default=3407, type=int, help='Fix the random seed for reproduction. Default is 25.')
    parser.add_argument('--use_warmup', type=bool, default=True)
    parser.add_argument('--warmup_epochs', default=10, type=float, help='warmup 阶段')
    parser.add_argument('--final_div_factor', default=1e5, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--DistributionSampler', type=bool, default=True)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./resnet50-pre.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()

    main(opt)
