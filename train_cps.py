import os
import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler


from utils import train_one_epoch, evaluate, load_data_minerals, try_gpu, get_priority


def main(args):
    device = try_gpu()  # 尝试使用 GPU
    print(args)
    print("运算设备为 {}".format(device))

    if os.path.exists("./weights") is False:  # 设置权重记录文件夹
        os.makedirs("./weights")

    if os.path.exists("./runs/logs") is False:  # 设置 tensorboard 记录文件夹
        os.makedirs("./runs/logs")

    tb_writer = SummaryWriter(log_dir="runs/logs")  # 实例化 SummaryWriter 对象

    # 加载矿物数据集
    train_loader, validate_loader = load_data_minerals(batch_size=args.batch_size,
                                                       need_index_train=True,
                                                       sampler="ClassPrioritySampler")

    model = models.resnet50()  # 定义模型

    if args.weights != "":  # 预训练权重不为空则进行载入
        assert os.path.exists(args.weights), "文件 {} 不存在".format(args.weights)
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    if args.freeze_layers:  # 冻结权重
        print("开启冻结")
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)  # 除 fc 外，其他权重全部冻结
            else:
                print("仅训练 {}".format(name))

    model.fc = nn.Linear(model.fc.in_features, args.num_classes)  # 修改最后一层
    nn.init.kaiming_normal_(model.fc.weight, nonlinearity='relu')  # 对 fc 层的权重进行 Kaiming 初始化
    nn.init.constant_(model.fc.bias, 0)  # 将 fc 层的偏置初始化为0

    model.to(device)

    tb_writer.add_graph(model, torch.zeros((1, 3, 224, 224), device=device))  # 将模型结构写入 tensorboard

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-5)  # 优化器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)  # 使用余弦衰减

    loss_function = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失

    best_acc = 0.0
    weight_list = []
    for epoch in range(args.epochs):
        # 训练

        train_loss, train_acc = train_one_epoch(model=model,
                                                loss_function=loss_function,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                end_batch=int(len(train_loader.dataset) / args.batch_size))

        if hasattr(train_loader.sampler, 'get_weights'):
            weight_list.append({key: value.tolist() for key, value in train_loader.sampler.get_weights().items()})
        if hasattr(train_loader.sampler, 'reset_weights'):
            train_loader.sampler.reset_weights(epoch)

        scheduler.step()

        # 验证
        val_loss, val_acc, total_logits, total_labels = evaluate(model=model,
                                                                 loss_function=loss_function,
                                                                 data_loader=validate_loader,
                                                                 device=device,
                                                                 epoch=epoch,
                                                                 need_val_info=True)

        if hasattr(train_loader.sampler, 'reset_priority'):
            ws = get_priority(total_logits.detach(), total_labels)
            train_loader.sampler.reset_priority(ws, total_labels.cpu().numpy())

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

    if hasattr(train_loader.sampler, 'get_weights'):
        with open('weight_list.pkl', 'wb') as file:
            pickle.dump(weight_list, file)
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.042)
    parser.add_argument('--eta-min', type=float, default=0.00042)
    parser.add_argument('--num_samples_cls', type=int, default=4)
    parser.add_argument('--pth_save_frequency', type=int, default=10)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./resnet50-pre.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()

    main(opt)
