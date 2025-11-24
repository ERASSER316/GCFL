import argparse
import copy
import csv
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import trange

import sys

# 保持导入路径不变
from experiments.GCFL.Models.CNNs import CNN_1,
from experiments.GCFL.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" ## 仅使用device2

random.seed(2022)

def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int, fraction: float,
          steps: int, epochs: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int) -> None:

    ###############################
    # init nodes                  #
    ###############################
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      batch_size=bs)

    # -------compute aggregation weights (Optional for Standalone)-------------#
    train_sample_count = nodes.train_sample_count
    eval_sample_count = nodes.eval_sample_count
    test_sample_count = nodes.test_sample_count

    client_sample_count = [train_sample_count[i] + eval_sample_count[i] + test_sample_count[i] for i in
                           range(len(train_sample_count))]
    # -----------------------------------------------#

    print(f"Dataset: {data_name} | Mode: Homogeneous Standalone | Model: CNN_1")

    # --- 修改点1：统一使用同构模型 (CNN_1) ---
    if data_name == "cifar10":
        net_template = CNN_1(n_kernels=n_kernels)
    elif data_name == "cifar100":
        net_template = CNN_1(n_kernels=n_kernels, out_dim=100)
    elif data_name == "mnist":
        net_template = CNN_1_MNIST()
    elif data_name == "fashion-mnist":
        net_template = CNN_1_MNIST()
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100', 'mnist', 'fashion-mnist']")

    net_template = net_template.to(device)
    
    # 只需要一个模板模型用于初始化
    # net_set 列表不再需要，因为所有人都一样

    ################
    # init metrics #
    ################
    criteria = torch.nn.CrossEntropyLoss()
    step_iter = trange(steps)

    PM_acc = defaultdict()
    PMs = defaultdict()
    
    # 初始化每个客户端的独立模型参数
    for i in range(num_nodes):
        PM_acc[i] = -1
        # 所有客户端从相同的初始权重开始（或者您可以去掉copy直接重新初始化以获得不同随机起点，通常保持一致便于对比）
        PMs[i] = copy.deepcopy(net_template.state_dict())

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # --- 修改点2：更新文件名 ---
    csv_file = str(save_path / f"Homo_Standalone_N{num_nodes}_C{fraction}_R{steps}_E{epochs}_B{bs}_NonIID_{data_name}_low_0.4.csv")
    
    with open(csv_file, 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')

        for r in step_iter:  # step is round
            frac = fraction
            select_nodes = random.sample(range(num_nodes), int(frac * num_nodes))

            all_local_trained_loss = []
            all_local_trained_acc = []
            
            # Standalone 模式下，Global Acc 通常没有意义，或者等同于 Local Acc 的平均
            # 这里为了保持 CSV 格式兼容，我们填入 0 或平均值
            all_global_loss = [] 
            all_global_acc = []
            results = []

            logging.info(f'#----Round:{r}----#')
            for c in select_nodes:
                node_id = c

                # 加载该客户端的模型
                # 使用模板重新加载参数，避免引用问题
                net = copy.deepcopy(net_template) 
                net.load_state_dict(PMs[node_id])
                net = net.to(device)

                # --- 修改点3：优化器必须在循环内初始化 ---
                # 确保每个客户端的动量状态是独立的，不共享
                if optim == 'sgd':
                    optimizer = torch.optim.SGD(params=net.parameters(), lr=inner_lr, momentum=0.9, weight_decay=wd)
                else:
                    optimizer = torch.optim.Adam(params=net.parameters(), lr=inner_lr)

                # Standalone 模式下，Global Eval 可以跳过，或者评估当前模型在测试集上的表现
                global_loss = 0
                global_acc = 0
                all_global_loss.append(global_loss)
                all_global_acc.append(global_acc)

                # Local Training
                net.train()
                for i in range(epochs):
                    for j, batch in enumerate(nodes.train_loaders[node_id], 0):
                        img, label = tuple(t.to(device) for t in batch)

                        optimizer.zero_grad()
                        pred, _ = net(img) # 注意：这里假设 CNN_1 返回 (pred, features)
                        loss = criteria(pred, label)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
                        optimizer.step()

                # Save local state
                PMs[node_id] = copy.deepcopy(net.state_dict())

                # Evaluate trained local model
                net.eval()
                with torch.no_grad():
                    test_acc = 0
                    num_batch = 0
                    test_loss = 0 # 补充 loss 计算

                    for batch in nodes.test_loaders[node_id]:
                        num_batch += 1
                        img, label = tuple(t.to(device) for t in batch)
                        pred, _ = net(img)
                        t_loss = criteria(pred, label)
                        test_loss += t_loss.item()
                        test_acc += pred.argmax(1).eq(label).sum().item() / len(label)

                    mean_test_loss = test_loss / num_batch
                    mean_test_acc = test_acc / num_batch

                all_local_trained_loss.append(mean_test_loss)
                all_local_trained_acc.append(mean_test_acc)
                PM_acc[node_id] = mean_test_acc

                logging.info(f'Round {r} | client {node_id} acc: {PM_acc[node_id]}')

            mean_trained_loss = round(np.mean(all_local_trained_loss), 4)
            mean_trained_acc = round(np.mean(all_local_trained_acc), 4)
            mean_global_loss = 0 # Placeholder
            mean_global_acc = 0  # Placeholder
            
            results.append([mean_global_loss, mean_global_acc, mean_trained_loss, mean_trained_acc] + [round(i,4) for i in PM_acc.values()])
            mywriter.writerows(results)
            file.flush()

            logging.info(f'Round:{r} | mean_trained_loss:{mean_trained_loss} | mean_trained_acc:{mean_trained_acc}')

            # --- 修改点4：完全移除聚合逻辑 ---
            # Standalone 模式下，不进行权重聚合，直接进入下一轮

        logging.info('Homogeneous Standalone Training finished successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Homogeneous Standalone Experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="fashion-mnist", choices=['cifar10', 'cifar100', 'mnist', 'fashion-mnist'], help="dir path for dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for dataset")
    parser.add_argument("--num-nodes", type=int, default=100, help="number of simulated nodes")
    parser.add_argument("--fraction", type=int, default=0.1, help="number of sampled nodes in each round")


    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=500, help='total number of rounds')
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer") 
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-3, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="Results/FedOFT_mh", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu) 

    if args.data_name == 'cifar10':
        args.classes_per_node = 10
    elif args.data_name == 'cifar100':
        args.classes_per_node = 100
    else:
        args.classes_per_node = 2

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        fraction = args.fraction,
        steps=args.num_steps,
        epochs=args.epochs,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed
    )