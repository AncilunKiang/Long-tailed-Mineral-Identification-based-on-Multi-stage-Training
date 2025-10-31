import os
import sys
import argparse
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler


from utils import evaluate, load_data_minerals, CenterLoss, try_gpu, compute_adjustment,\
    get_center_weight, update_env_by_score, ClassifierFC, CombinedModel


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

    env0_loader, = *load_data_minerals(batch_size=args.batch_size,
                                       max_num_workers=args.max_num_workers,
                                       pin_memory=True,
                                       need_index_train=True,
                                       sampler="DistributionSampler",
                                       batch_split=args.batch_split,
                                       just_env0=True,
                                       just_train=True),

    if args.need_logit_adj_loss:
        loader_for_num, = *load_data_minerals(batch_size=args.batch_size,
                                              max_num_workers=4,
                                              pin_memory=True,
                                              just_train=True),
    else:
        loader_for_num = None

    model = models.resnet50()  # 定义模型

    if args.weights != "":  # 预训练权重不为空则进行载入
        assert os.path.exists(args.weights), "文件 {} 不存在".format(args.weights)
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    if args.freeze_layers:  # 冻结权重
        print("开启冻结")
        for name, para in model.named_parameters():
            para.requires_grad_(False)

    feat_dim = model.fc.in_features
    if args.is_ReLU:
        model.fc = nn.ReLU()  # 原论文做法
    else:
        model.fc = torch.nn.Identity()  # 删除分类头

    model.to(device)

    classifier = ClassifierFC(feat_dim=feat_dim, num_classes=args.num_classes, is_bias=args.is_bias)  # 创建分类器

    nn.init.kaiming_normal_(classifier.fc.weight, nonlinearity='relu')  # 对 fc 层的权重进行 Kaiming 初始化
    if args.is_bias:
        nn.init.constant_(classifier.fc.bias, 0)  # 将 fc 层的偏置初始化为0
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

    env1_loader, env2_loader = train_loader  # 双环境
    assert len(env1_loader) == len(env2_loader)
    total_image = len(env1_loader.dataset)
    # self.系列保存全局训练信息
    all_ind_self = []
    all_lab_self = []
    all_prb_self = []
    all_lgt_self = []

    weight_list = [(env2_loader.sampler.get_weight()).tolist()]  # 记录采样权重

    if args.need_logit_adj_loss:
        logit_adjustments = compute_adjustment(loader_for_num, args.tro_train, device)
    else:
        logit_adjustments = None
    del loader_for_num

    best_acc = 0.0
    for epoch in range(args.update_milestones[0]+1):  # 训练
        # 为环境建设保存训练信息
        all_ind = []
        all_lab = []
        all_prb = []
        all_lgt = []

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
            indexs = indexs.to(device)

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

            # 保存环境分割信息
            all_lgt.append(pred.detach().clone().cpu())
            gt_score = torch.gather(pred.softmax(-1), 1, torch.unsqueeze(labels, 1)).view(-1)
            all_ind.append(indexs.detach().clone().cpu())
            all_lab.append(labels.detach().clone().cpu())
            all_prb.append(gt_score.detach().clone().cpu())

            # 计算本轮次的平均损失
            train_loss = accu_loss.item() / (step + 1)
            # 计算本轮次的精度
            train_acc = accu_num.item() / sample_num
            # 打到进度条上
            pbar.set_description("[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, train_loss, train_acc))
            pbar.update()

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                pbar.close()
                sys.exit(1)
        pbar.close()

        env_score_memo = {}
        if epoch in args.update_milestones:
            print("更新环境 2")
            # update env mask
            all_ind_self = torch.cat(all_ind, dim=0)
            all_lab_self = torch.cat(all_lab, dim=0)
            all_prb_self = torch.cat(all_prb, dim=0)
            all_lgt_self = torch.cat(all_lgt, dim=0)

            print(f'Saving all_***_self_{epoch}')
            with open(f'all_ind_self_{epoch}.pkl', 'wb') as file:
                pickle.dump(all_ind_self, file)
            with open(f'all_lab_self_{epoch}.pkl', 'wb') as file:
                pickle.dump(all_lab_self, file)
            with open(f'all_prb_self_{epoch}.pkl', 'wb') as file:
                pickle.dump(all_prb_self, file)
            with open(f'all_lgt_self_{epoch}.pkl', 'wb') as file:
                pickle.dump(all_lgt_self, file)
            print('Finished Saving all_***_self_{epoch}')

            # save env_score
            env_score_memo['label_{}'.format(epoch)] = all_lab_self.tolist()
            env_score_memo['prob_{}'.format(epoch)] = all_prb_self.tolist()
            env_score_memo['idx_{}'.format(epoch)] = all_ind_self.tolist()

            update_env_by_score(all_ind_self, all_lab_self, all_prb_self,
                                env1_loader, env2_loader,
                                total_image, args.sample_scale)

            weight_list.append((env2_loader.sampler.get_weight()).tolist())

        scheduler.step()

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
        if val_acc >= best_acc:  # 每 args.pth_save_frequency 轮保存最佳模型
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(int(epoch / args.pth_save_frequency)))
            torch.save(classifier.state_dict(), "./weights/classifier-{}.pth".format(int(epoch / args.pth_save_frequency)))

    torch.save(model.state_dict(), "./weights/model-{}.pth".format(args.update_milestones[0]))
    torch.save(classifier.state_dict(), "./weights/classifier-{}.pth".format(args.update_milestones[0]))

    del env0_loader

    for epoch in range(args.update_milestones[0]+1, args.epochs):  # 训练
        # 为环境建设保存训练信息
        all_ind = []
        all_lab = []
        all_prb = []
        all_lgt = []

        center_weight = get_center_weight(epoch, args.center_weights, args.center_milestones)

        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_loss_ce = torch.zeros(1).to(device)  # 累计主损失
        accu_loss_ct = torch.zeros(1).to(device)  # 累计中心损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        accu_num_env1 = torch.zeros(1).to(device)  # 累计环境 1 预测正确的样本数
        accu_num_env2 = torch.zeros(1).to(device)  # 累计环境 2 预测正确的样本数

        sample_num = 0
        train_loss = train_loss_ce = train_loss_ct = 0.0  # 总损失 主损失 中心损失
        train_acc = train_acc_env1 = train_acc_env2 = 0.0  # 总精度 环境1精度 环境2精度
        pbar = tqdm(total=len(env1_loader), file=sys.stdout)  # 设置手动进度条
        for step, ((inputs1, labels1, indexs1), (inputs2, labels2, indexs2)) in enumerate(zip(env1_loader, env2_loader)):
            model.train()
            classifier.train()
            optimizer.zero_grad()

            # 拼接双环境
            inputs = torch.cat([inputs1, inputs2], dim=0).to(device)
            labels = torch.cat([labels1, labels2], dim=0).to(device)
            indexs = torch.cat([indexs1, indexs2], dim=0).to(device)

            sample_num += inputs.shape[0]  # 样本量计数（双环境，所以如果不开 batch_split 的话一个 batch 是 batch_size 的 2 倍）

            features = model(inputs)  # 特征
            pred = classifier(features)  # logits，add_inputs 在这儿没啥用

            # 计数各种预测正确的样本数
            pred_classes = torch.max(pred, dim=1)[1]  # 获取预测值
            accu_num += torch.eq(pred_classes, labels).sum()  # 累计预测正确的样本数
            # 累计环境1（前半拉）预测正确的样本数
            accu_num_env1 += torch.eq(pred_classes[:inputs1.shape[0]], labels1.to(device)).sum()
            # 累计环境2（后半拉）预测正确的样本数
            accu_num_env2 += torch.eq(pred_classes[inputs1.shape[0]:], labels2.to(device)).sum()

            if args.need_logit_adj_loss:  # 如果设置了 logits 调整损失则进行调整
                pred += logit_adjustments

            # 计算损失们
            loss_ce = loss_function(pred, labels)  # 计算主损失
            if args.need_logit_adj_loss:  # 如果设置了 logits 调整损失则进行调整
                loss_r = 0
                for parameter in model.parameters():  # 对每个模型参数
                    loss_r += torch.sum(parameter ** 2)  # 计算L2正则化损失
                for parameter in classifier.parameters():  # 对每个模型参数
                    loss_r += torch.sum(parameter ** 2)  # 计算L2正则化损失
                loss_ce += args.weight_decay * loss_r  # 添加正则化损失

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

            # 保存环境分割信息
            all_lgt.append(pred.detach().clone().cpu())
            gt_score = torch.gather(pred.softmax(-1), 1, torch.unsqueeze(labels, 1)).view(-1)
            all_ind.append(indexs.detach().clone().cpu())
            all_lab.append(labels.detach().clone().cpu())
            all_prb.append(gt_score.detach().clone().cpu())

            # 计算本轮次的各种平均损失
            train_loss = accu_loss.item() / (step + 1)
            train_loss_ce = accu_loss_ce.item() / (step + 1)
            train_loss_ct = accu_loss_ct.item() / (step + 1)
            # 计算本轮次的各种精度
            train_acc = accu_num.item() / sample_num
            train_acc_env1 = accu_num_env1.item() / (sample_num // 2)
            train_acc_env2 = accu_num_env2.item() / (sample_num // 2)
            # 打到进度条上
            pbar.set_description("[train epoch {}] loss: {:.3f}, loss_ce: {:.3f}, loss_ct: {:.3f}, acc: {:.3f}, "
                                 "acc_env1: {:.3f}, acc_env2: {:.3f}".format(epoch,
                                                                             train_loss, train_loss_ce, train_loss_ct,
                                                                             train_acc, train_acc_env1, train_acc_env2))
            pbar.update()

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                pbar.close()
                sys.exit(1)
        pbar.close()

        env_score_memo = {}
        if epoch in args.update_milestones:
            print("更新环境 2")
            # update env mask
            all_ind_self = torch.cat(all_ind, dim=0)
            all_lab_self = torch.cat(all_lab, dim=0)
            all_prb_self = torch.cat(all_prb, dim=0)
            all_lgt_self = torch.cat(all_lgt, dim=0)

            print(f'Saving all_***_self_{epoch}')
            with open(f'all_ind_self_{epoch}.pkl', 'wb') as file:
                pickle.dump(all_ind_self, file)
            with open(f'all_lab_self_{epoch}.pkl', 'wb') as file:
                pickle.dump(all_lab_self, file)
            with open(f'all_prb_self_{epoch}.pkl', 'wb') as file:
                pickle.dump(all_prb_self, file)
            with open(f'all_lgt_self_{epoch}.pkl', 'wb') as file:
                pickle.dump(all_lgt_self, file)
            print('Finished Saving all_***_self_{epoch}')

            # save env_score
            env_score_memo['label_{}'.format(epoch)] = all_lab_self.tolist()
            env_score_memo['prob_{}'.format(epoch)] = all_prb_self.tolist()
            env_score_memo['idx_{}'.format(epoch)] = all_ind_self.tolist()

            update_env_by_score(all_ind_self, all_lab_self, all_prb_self,
                                env1_loader, env2_loader,
                                total_image, args.sample_scale)

            weight_list.append((env2_loader.sampler.get_weight()).tolist())

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
        tb_writer.add_scalar("train_acc_env1", train_acc_env1, epoch)
        tb_writer.add_scalar("train_acc_env2", train_acc_env2, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if epoch % args.pth_save_frequency == 0:  # 每 args.pth_save_frequency 轮 best_acc 清零 保存一次模型
            best_acc = 0.0
        if val_acc >= best_acc:  # 每 args.pth_save_frequency 轮保存最佳模型
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(int(epoch / args.pth_save_frequency)))
            torch.save(classifier.state_dict(), "./weights/classifier-{}.pth".format(int(epoch / args.pth_save_frequency)))

    torch.save(model.state_dict(), "./weights/model-end.pth")
    torch.save(classifier.state_dict(), "./weights/classifier-end.pth")

    print('Finished Training')

    print('Saving weight_list')
    with open('weight_list.pkl', 'wb') as file:
        pickle.dump(weight_list, file)
    print('Finished Saving weight_list')

    print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)  # 256
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--eta-min', type=float, default=0.0)  # 0.00042
    parser.add_argument('--pth_save_frequency', type=int, default=5)
    parser.add_argument('--batch_split', help='环境 batch 分割', type=bool, default=False)
    parser.add_argument('--max_num_workers', help='最大加载线程数', type=int, default=4)
    parser.add_argument('--center_weights', help='折衷参数 α', type=list, default=[0.0, 0.003, 0.005])  # [0.0, 0.001, 0.005]
    parser.add_argument('--center_milestones', help='折衷参数跳变点', type=list, default=[0, 60, 80])
    parser.add_argument('--update_milestones', help='环境更新点', type=list, default=[60, 80])  # [60, 80]
    parser.add_argument('--sample_scale', help='采样间隔', type=float, default=4.0)
    parser.add_argument('--seed', default=3407, type=int, help='Fix the random seed for reproduction. Default is 25.')
    parser.add_argument('--lr_ct', type=float, default=0.5)  # 0.5
    parser.add_argument('--is_bias', help='分类头是否带偏置', type=bool, default=True)
    parser.add_argument('--is_ReLU', help='是否换 relu', type=bool, default=False)
    parser.add_argument('--need_logit_adj_loss', help='是否开启 logits 调整损失', type=bool, default=True)
    parser.add_argument('--tro_train', default=1.0, type=float, help='tro for logit adj train')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        default='',
                        # default='./resnet50-pre.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()

    main(opt)
