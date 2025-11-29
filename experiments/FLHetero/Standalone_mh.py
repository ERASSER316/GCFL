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
#sys.path.append('/home/ylp/HN/pFedHN-main/')

from experiments.GCFL.Models.CNNs import CNN_1,CNN_2,CNN_3,CNN_4,CNN_5, CNN_1_MNIST,CNN_2_MNIST,CNN_3_MNIST,CNN_4_MNIST,CNN_5_MNIST
from experiments.GCFL.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" ## 仅使用device0和 device1

random.seed(2022)



def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int, fraction: float,
          steps: int, epochs: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int) -> None:

    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      batch_size=bs)

    # -------compute aggregation weights-------------#
    train_sample_count = nodes.train_sample_count
    eval_sample_count = nodes.eval_sample_count
    test_sample_count = nodes.test_sample_count

    client_sample_count = [train_sample_count[i] + eval_sample_count[i] + test_sample_count[i] for i in
                           range(len(train_sample_count))]
    # -----------------------------------------------#


    print(data_name)
    if data_name == "cifar10":
        net_1 = CNN_1(n_kernels=n_kernels)
        net_2 = CNN_2(n_kernels=n_kernels)
        net_3 = CNN_3(n_kernels=n_kernels)
        net_4 = CNN_4(n_kernels=n_kernels)
        net_5 = CNN_5(n_kernels=n_kernels)
    elif data_name == "cifar100":
        net_1 = CNN_1(n_kernels=n_kernels, out_dim=100)
        net_2 = CNN_2(n_kernels=n_kernels, out_dim=100)
        net_3 = CNN_3(n_kernels=n_kernels, out_dim=100)
        net_4 = CNN_4(n_kernels=n_kernels, out_dim=100)
        net_5 = CNN_5(n_kernels=n_kernels, out_dim=100)
    elif data_name == "mnist":
        net_1 = CNN_1_MNIST()
        net_2 = CNN_2_MNIST()
        net_3 = CNN_3_MNIST()
        net_4 = CNN_4_MNIST()
        net_5 = CNN_5_MNIST()
    elif data_name == "fashion-mnist":
        net_1 = CNN_1_MNIST()
        net_2 = CNN_2_MNIST()
        net_3 = CNN_3_MNIST()
        net_4 = CNN_4_MNIST()
        net_5 = CNN_5_MNIST()
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")

    net_1 = net_1.to(device)
    net_2 = net_2.to(device)
    net_3 = net_3.to(device)
    net_4 = net_4.to(device)
    net_5 = net_5.to(device)

    net_set = [net_1, net_2, net_3, net_4, net_5]



    ##################
    # init optimizer #
    ##################
    # net = net_1
    #
    # optimizers = {
    #     'sgd': torch.optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=wd),
    #     'adam': torch.optim.Adam(params=net.parameters(), lr=lr)
    # }
    # optimizer = optimizers[optim]
    criteria = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
    #                                            milestones=[int(steps * 0.56), int(steps * 0.78)],
    #                                            gamma=0.1, last_epoch=-1)

    ################
    # init metrics #
    ################
    step_iter = trange(steps)

    PM_acc = defaultdict()
    PMs = defaultdict()
    for i in range(num_nodes):
        PM_acc[i] = -1
        PMs[i] = copy.deepcopy(net_set[i%5].state_dict())


    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / f"Hetero_Standalone_N{num_nodes}_C{fraction}_R{steps}_E{epochs}_B{bs}_NonIID_{data_name}_low_0.4.csv"), 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')


        for r in step_iter:  # step is round [可调参数]
            frac = fraction #选中客户端的比例，[可调参数]
            select_nodes = random.sample(range(num_nodes), int(frac * num_nodes))


            all_local_trained_loss = []
            all_local_trained_acc = []
            all_global_loss = []
            all_global_acc = []
            results = []

            # LNs = defaultdict()
            logging.info(f'#----Round:{r}----#')
            for c in select_nodes:
                node_id = c

                net = net_set[node_id % 5]
                optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                net.load_state_dict(PMs[node_id])

                # evlaute GM
                # global_loss,  global_acc = test_acc(net, nodes.test_loaders[node_id], criteria)
                global_loss = 0
                global_acc = 0
                all_global_loss.append(global_loss)
                all_global_acc.append(global_acc)

                # local training

                net.train()
                for i in range(epochs):
                    for j, batch in enumerate(nodes.train_loaders[node_id], 0):
                        img, label = tuple(t.to(device) for t in batch)

                        optimizer.zero_grad()

                        pred, _ = net(img)
                        # print(f'pred:{pred}')
                        loss = criteria(pred, label)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
                        optimizer.step()
                    # print(f'Epoch:{i} | Loss:{loss}')
                #

                # # collect local NN parameters
                PMs[node_id] = copy.deepcopy(net.state_dict())

                # evaluate trained local model
                net.eval()
                with torch.no_grad():
                    test_acc = 0
                    num_batch = 0

                    for batch in nodes.test_loaders[node_id]:
                        num_batch += 1
                        # batch = next(iter(testloader))
                        img, label = tuple(t.to(device) for t in batch)
                        pred, _ = net(img)
                        test_loss = criteria(pred, label)
                        test_acc += pred.argmax(1).eq(label).sum().item() / len(label)

                    mean_test_loss = test_loss / num_batch
                    mean_test_acc = test_acc / num_batch


                all_local_trained_loss.append(mean_test_loss.cpu().item())
                all_local_trained_acc.append(mean_test_acc)
                PM_acc[node_id] = mean_test_acc

                logging.info(f'Round {r} | client {node_id} acc: {PM_acc[node_id]}')

            # # 更新学习率并查看当前学习率
            # scheduler.step()
            # # print('\t last_lr:', scheduler.get_last_lr())
            # logging.info(f'learning rate | {scheduler.get_last_lr()}')

            mean_trained_loss = round(np.mean(all_local_trained_loss), 4)
            mean_trained_acc = round(np.mean(all_local_trained_acc), 4)
            mean_global_loss = round(np.mean(all_global_loss), 4)
            mean_global_acc = round(np.mean(all_global_acc), 4)
            results.append([mean_global_loss, mean_global_acc, mean_trained_loss, mean_trained_acc] + [round(i,4) for i in PM_acc.values()])
            mywriter.writerows(results)
            file.flush()

            logging.info(f'Round:{r} | mean_global_loss:{mean_global_loss} | mean_global_acc:{mean_global_acc} | mean_trained_loss:{mean_trained_loss} | mean_trained_acc:{mean_trained_acc}')

            # client_agg_weights = OrderedDict()
            # select_nodes_sample_count = OrderedDict()
            # for i in range(len(select_nodes)):
            #     select_nodes_sample_count[select_nodes[i]] = client_sample_count[select_nodes[i]]
            # for i in range(len(select_nodes)):
            #     client_agg_weights[select_nodes[i]] = select_nodes_sample_count[select_nodes[i]] / sum(select_nodes_sample_count.values())
            #
            # # aggregate
            # weight_keys = list(net.state_dict().keys())
            # Gnet_paras = OrderedDict()
            # for key in weight_keys:
            #     key_sum = 0
            #     for id, model in LNs.items():
            #         key_sum += client_agg_weights[id] * model[key]
            #     Gnet_paras[key] = key_sum
            # logging.info(f'Global model is updated after aggregation')

        logging.info('Federated Learning has been successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Learning with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="fashion-mnist", choices=['cifar10', 'cifar100', 'mnist', 'fashion-mnist'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
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
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer") #[可调参数]
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

    device = get_device(gpus=args.gpu) #改gpu id

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
