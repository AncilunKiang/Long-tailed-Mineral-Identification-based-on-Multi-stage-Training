import os
import argparse

import torch
import random
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler


from utils import train_one_epoch, evaluate, try_gpu, load_data_minerals, compute_adjustment, ClassifierFC


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
                                                       sampler="ClassAwareSampler",
                                                       num_samples_cls=4,
                                                       max_num_workers=4,
                                                       pin_memory=True)

    if args.need_logit_adj_loss:
        loader_for_num, = *load_data_minerals(batch_size=args.batch_size,
                                              max_num_workers=4,
                                              pin_memory=True,
                                              just_train=True),
    else:
        loader_for_num = None

    model = models.resnet50()  # 定义模型
    feat_dim = model.fc.in_features

    if args.freeze_layers:  # 只有冻结时才会加载已经训练的参数，才需要对应的修改
        if args.weights_classifier != '' and args.is_ReLU:
            model.fc = nn.ReLU()  # 原论文做法
        elif args.weights_classifier != '' and not args.is_ReLU:
            model.fc = torch.nn.Identity()  # 删除分类头
        else:
            model.fc = nn.Linear(feat_dim, args.num_classes)  # 修改最后一层

    if args.weights != "":  # 预训练权重不为空则进行载入
        assert os.path.exists(args.weights), "文件 {} 不存在".format(args.weights)
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    if args.freeze_layers:  # 冻结权重
        print("开启冻结")
        if args.weights_classifier != '':  # 模型拆分则特征提取器全部冻结
            for name, para in model.named_parameters():
                para.requires_grad_(False)
            print("仅训练：['classifier.fc.weight', 'classifier.fc.bias']")
        else:
            trained_parameters_list = []
            for name, para in model.named_parameters():
                if "fc" not in name:
                    para.requires_grad_(False)  # 除 fc 外，其他权重全部冻结
                else:
                    trained_parameters_list.append('model.' + name)
            assert len(trained_parameters_list) > 0, "没有参数参与训练"
            print("仅训练：{}".format(trained_parameters_list))

    if not args.freeze_layers and args.weights_classifier == '':  # 不拆分不冻结则按正常来
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)  # 修改最后一层

    if args.weights_classifier != '':
        classifier = ClassifierFC(feat_dim=feat_dim, num_classes=args.num_classes, is_bias=args.is_bias)  # 创建分类器
        assert os.path.exists(args.weights_classifier), "文件 {} 不存在".format(args.weights_classifier)
        classifier.load_state_dict(torch.load(args.weights_classifier, map_location='cpu'))
    else:
        classifier = None

    if args.classifier_retrain:
        if args.weights_classifier != '':
            nn.init.kaiming_normal_(classifier.fc.weight, nonlinearity='relu')  # 对 fc 层的权重进行 Kaiming 初始化
            if args.is_bias:
                nn.init.constant_(classifier.fc.bias, 0)  # 将 fc 层的偏置初始化为0
        else:
            nn.init.kaiming_normal_(model.fc.weight, nonlinearity='relu')  # 对 fc 层的权重进行 Kaiming 初始化
            nn.init.constant_(model.fc.bias, 0)  # 将 fc 层的偏置初始化为0

    model.to(device)
    if classifier:
        classifier.to(device)

    tb_writer.add_graph(model, torch.zeros((1, 3, 224, 224), device=device))  # 将模型结构写入 tensorboard

    pg_model = [p for p in model.parameters() if p.requires_grad]  # 特征提取器需要训练的参数
    if args.weights_classifier != '':
        pg_classifier = [p for p in classifier.parameters() if p.requires_grad]
    else:
        pg_classifier = []
    optimizer = optim.SGD(pg_model + pg_classifier, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # 优化器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)  # 使用余弦衰减

    loss_function = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失

    if args.need_logit_adj_loss:
        logit_adjustments = compute_adjustment(loader_for_num, args.tro_train, device)
    else:
        logit_adjustments = None
    del loader_for_num

    best_acc = 0.0
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model=model,
                                                loss_function=loss_function,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                end_batch=int(len(train_loader.dataset) / args.batch_size + 0.5),
                                                do_shuffle=args.do_shuffle,
                                                logit_adjustments=logit_adjustments,
                                                weight_decay=args.weight_decay,
                                                classifier=classifier)

        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model=model,
                                     loss_function=loss_function,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch,
                                     classifier=classifier)

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
            if classifier:
                torch.save(classifier.state_dict(),
                           "./weights/classifier-{}.pth".format(int(epoch / args.pth_save_frequency)))

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eta-min', type=float, default=0.000000001)
    parser.add_argument('--num_samples_cls', type=int, default=4)
    parser.add_argument('--pth_save_frequency', type=int, default=1)
    parser.add_argument('--do_shuffle', type=bool, default=True)
    parser.add_argument('--classifier_retrain', type=bool, default=False)
    parser.add_argument('--need_logit_adj_loss', help='是否开启 logits 调整损失', type=bool, default=True)
    parser.add_argument('--tro_train', default=1.0, type=float, help='tro for logit adj train')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--is_bias', help='分类头是否带偏置', type=bool, default=True)
    parser.add_argument('--is_ReLU', help='是否换 relu', type=bool, default=False)
    parser.add_argument('--seed', default=3407, type=int, help='Fix the random seed for reproduction. Default is 25.')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        # default='./resnet50-pre.pth',
                        # default='./model-9-CPS.pth',
                        default='model-end-lal-006.pth',
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/ResNet50/008（次序正常版）/weights'
                        #         '/model-9.pth',
                        help='initial weights path')

    parser.add_argument('--weights_classifier', type=str,
                        # default='',
                        # default='C:/Users/AncilunKiang/Desktop/DesktopFile/研究生日常/炼丹日志/cls/ifl/020/weights/classifier'
                        #         '-15.pth',
                        default='./classifier-end-lal-006.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)

    opt = parser.parse_args()

    main(opt)
