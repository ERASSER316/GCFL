import argparse
import copy
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import trange

from experiments.GCFL.Models.CNNs import CNN_1, GC_CNN_1
from experiments.GCFL.Models.GCBlock import (
    GCBlock,
    build_fused_state_dict,
    load_fused_weights_into_gc_model,
)
from experiments.GCFL.node import BaseNodes
from experiments.utils import (
    format_num_bytes,
    get_device,
    set_logger,
    set_seed,
    state_dict_num_bytes,
    state_dict_parameter_count,
    str2bool,
)

# Removed global random.seed(2022) to unify seed management in main

def build_model(data_name: str, n_kernels: int, model_name: str, num_classes: int, device: torch.device):
    # Logic updated to accept num_classes explicitly
    in_channels = 1 if data_name in {"mnist", "fashion-mnist"} else 3
    out_dim = num_classes

    if model_name == "gc_cnn1":
        model = GC_CNN_1(in_channels=in_channels, n_kernels=n_kernels, out_dim=out_dim)
    elif model_name == "cnn1":
        model = CNN_1(in_channels=in_channels, n_kernels=n_kernels, out_dim=out_dim)
    else:
        raise ValueError(f"Unsupported model '{model_name}'")

    return model.to(device)

#寻找模型中所有GCBlock模块的名称，可以在模型构建的时候进行不止一处的gcblock融合
def get_gcblock_prefixes(model: torch.nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, GCBlock)]


def is_gcblock_key(key: str, prefixes: List[str]) -> bool:
    return any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)


def fed_avg_state_dicts(state_dicts: List[Dict], weights: List[float]) -> Dict: #传入多个客户端模型的列表和对应的样本数（权重）
    total_weight = float(sum(weights))
    averaged: dict[str, torch.Tensor] = {}

    for key in state_dicts[0].keys():
        weighted_sum = None
        for sd, w in zip(state_dicts, weights):
            value = sd[key].float()
            scaled = value * (w / total_weight)
            weighted_sum = scaled if weighted_sum is None else weighted_sum + scaled
        averaged[key] = weighted_sum
    return averaged


def unpack_prediction(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def train(
    data_name: str,
    data_path: str,
    classes_per_node: int,
    num_nodes: int,
    fraction: float,
    steps: int,
    epochs: int,
    optim: str,
    lr: float,
    inner_lr: float,
    embed_lr: float,
    wd: float,
    inner_wd: float,
    embed_dim: int,
    hyper_hid: int,
    n_hidden: int,
    n_kernels: int,
    bs: int,
    device,
    eval_every: int,
    save_path: Path,
    seed: int,
    model_name: str,
    use_gcblock: bool,
    gcfl_variant: str,
    log_comm_to_csv: bool,
) -> None:
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node, batch_size=bs)

    train_sample_count = nodes.train_sample_count
    eval_sample_count = nodes.eval_sample_count
    test_sample_count = nodes.test_sample_count

    #统计每个客户端的样本总数（该客户端的训练样本数+验证样本数+测试样本数），用于后面聚合时候的加权平均
    client_sample_count = [
        train_sample_count[i] + eval_sample_count[i] + test_sample_count[i]
        for i in range(len(train_sample_count))
    ]

    # Determine total classes for model building
    # Assuming standard datasets
    if data_name == "cifar100":
        total_num_classes = 100
    else:
        total_num_classes = 10
    #根据shell输入构建CNN/GC-CNN模型
    net_template = build_model(data_name, n_kernels, model_name, total_num_classes, device)
    gcblock_prefixes = get_gcblock_prefixes(net_template)

    logging.info(
        "Dataset: %s | Mode: Homogeneous Standalone | Model: %s | GCBlock: %s",
        data_name,
        model_name,
        use_gcblock,
    )

    criteria = torch.nn.CrossEntropyLoss()
    step_iter = trange(steps)   #for r in range(steps)的带进度条版本，steps是联邦学习的总轮数

    # PM_acc keeps track of the last known accuracy of each client for logging
    PM_acc = defaultdict(float) #用于存储每个客户端的最新验证准确率，key是客户端索引，value是准确率
    for i in range(num_nodes):  #初始化每个客户端的验证准确率为-1.0
        PM_acc[i] = -1.0
    
    # Memory Optimization: Removed PMs dictionary. Use a single global state.
    global_state_dict = copy.deepcopy(net_template.state_dict()) #全局模型参数

    communication_params = [] #用于存储每轮通信的参数数量
    communication_bytes = [] #用于存储每轮通信的字节数

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    csv_file = str(
        save_path
        / f"Homo_Standalone_N{num_nodes}_C{fraction}_R{steps}_E{epochs}_B{bs}_NonIID_{data_name}_low_0.4.csv"
    )

    with open(csv_file, "w", newline="") as file:
        mywriter = csv.writer(file, delimiter=",")
        
        # Code Quality: Add CSV Header，记录训练数据更新
        headers = ["Global Loss", "Global Acc", "Mean Local Loss", "Mean Local Acc"]
        headers.extend([f"Client_{i}_Acc" for i in range(num_nodes)])
        if log_comm_to_csv:
            headers.extend(["Round Params", "Round Bytes", "Avg Params", "Avg Bytes"])
        mywriter.writerow(headers)

        for r in step_iter:  # step is round
            frac = fraction
            select_nodes = random.sample(range(num_nodes), int(frac * num_nodes))   #select_nodes是一个列表，包含了本次轮次中被选中的客户端索引

            logging.info("#----Round:%s----#", r)

            # --- Critical Fix: Global Evaluation ---
            # Calculate Global Accuracy BEFORE training
            net_template.load_state_dict(global_state_dict)
            net_template.eval()
            
            total_eval_loss = 0
            total_eval_acc = 0
            total_eval_samples = 0
            
            # Evaluate on a subset or all test loaders to get Global Performance
            # Here we iterate all test loaders to get accurate global stats
            with torch.no_grad():
                for node_id in range(num_nodes):    #遍历所有客户端
                    for batch in nodes.test_loaders[node_id]:    #遍历当前客户端的测试集
                        img, label = tuple(t.to(device) for t in batch)    #将当前客户端的测试集数据转换为tensor
                        pred = unpack_prediction(net_template(img))    #通过模型预测当前客户端的测试集数据
                        loss = criteria(pred, label)    #计算损失
                        total_eval_loss += loss.item() * len(label)    #累加损失
                        total_eval_acc += pred.argmax(1).eq(label).sum().item()    #累加正确预测数
                        total_eval_samples += len(label)    #累加样本数
            
            global_loss = total_eval_loss / total_eval_samples if total_eval_samples > 0 else 0 #计算全局损失
            global_acc = total_eval_acc / total_eval_samples if total_eval_samples > 0 else 0 #计算全局准确率
            
            # ---------------------------------------

            all_local_trained_loss = [] #用于存储每个客户端的训练损失
            all_local_trained_acc = [] #用于存储每个客户端的训练准确率
            
            fused_states = [] #用于存储每轮通信的融合状态
            non_gc_states = [] #用于存储每轮通信的非融合状态
            selected_weights = [] #用于存储每轮通信的权重

            round_param_count = 0 #用于存储每轮通信的参数数量
            round_byte_count = 0 #用于存储每轮通信的字节数

            for c in select_nodes:  #遍历本次轮次中被选中的fraction比例的客户端，c是客户端索引
                node_id = c

                net = copy.deepcopy(net_template)
                # Memory Optimization: Load from global state, not PMs
                net.load_state_dict(global_state_dict) #初始化当前客户端的模型参数为全局模型参数
                net = net.to(device)

                if optim == "sgd":
                    optimizer = torch.optim.SGD(params=net.parameters(), lr=inner_lr, momentum=0.9, weight_decay=wd)
                else:
                    optimizer = torch.optim.Adam(params=net.parameters(), lr=inner_lr)

                net.train()
                for _ in range(epochs):
                    for batch in nodes.train_loaders[node_id]:
                        img, label = tuple(t.to(device) for t in batch)
                        optimizer.zero_grad()
                        pred = unpack_prediction(net(img))
                        loss = criteria(pred, label)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)    #梯度裁剪，防止梯度爆炸
                        optimizer.step()

                # Local Evaluation
                net.eval()
                with torch.no_grad():
                    test_acc = 0
                    num_batch = 0
                    test_loss = 0

                    for batch in nodes.test_loaders[node_id]:
                        num_batch += 1
                        img, label = tuple(t.to(device) for t in batch)
                        pred = unpack_prediction(net(img))
                        t_loss = criteria(pred, label)
                        test_loss += t_loss.item()
                        test_acc += pred.argmax(1).eq(label).sum().item() / len(label)

                    mean_test_loss = test_loss / num_batch
                    mean_test_acc = test_acc / num_batch

                all_local_trained_loss.append(mean_test_loss)
                all_local_trained_acc.append(mean_test_acc)
                PM_acc[node_id] = mean_test_acc

                logging.info("Round %s | client %s acc: %s", r, node_id, PM_acc[node_id])

                if use_gcblock:
                    fused_state = build_fused_state_dict(net)
                    base_state = {
                        k: v.detach().clone()
                        for k, v in net.state_dict().items()
                        if not is_gcblock_key(k, gcblock_prefixes)
                    }

                    fused_states.append(fused_state)
                    non_gc_states.append(base_state)
                    selected_weights.append(client_sample_count[node_id])
                    #算这轮该客户端上传的参数个数和字节数
                    upload_param_count = state_dict_parameter_count(fused_state) + state_dict_parameter_count(base_state)
                    upload_byte_count = state_dict_num_bytes(fused_state) + state_dict_num_bytes(base_state)
                else:
                    full_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    
                    # For standard FedAvg, we treat full state as non_gc_state for unified aggregation logic below
                    # or aggregate separately. Here we fit into the lists:
                    non_gc_states.append(full_state) # Treat all as base
                    selected_weights.append(client_sample_count[node_id])
                    
                    upload_param_count = state_dict_parameter_count(full_state)
                    upload_byte_count = state_dict_num_bytes(full_state)
                #这两个在所有选中客户端上累加，得到这一轮所有客户端总上传量
                round_param_count += upload_param_count
                round_byte_count += upload_byte_count

            mean_trained_loss = round(np.mean(all_local_trained_loss), 4)
            mean_trained_acc = round(np.mean(all_local_trained_acc), 4)
            
            # Prepare results row
            row = [global_loss, global_acc, mean_trained_loss, mean_trained_acc]
            row.extend([round(PM_acc[i], 4) for i in range(num_nodes)])

            if round_byte_count:
                communication_params.append(round_param_count)
                communication_bytes.append(round_byte_count)

                avg_params = int(np.mean(communication_params))
                avg_bytes = float(np.mean(communication_bytes))

                logging.info(
                    "Round:%s | upload_params:%s | upload_bytes:%s | avg_upload_bytes:%s",
                    r,
                    round_param_count,
                    format_num_bytes(round_byte_count),
                    format_num_bytes(avg_bytes),
                )

                if log_comm_to_csv:
                    row.extend([round_param_count, round_byte_count, avg_params, int(avg_bytes)])
            
            results = [row]
            mywriter.writerows(results)
            file.flush()

            logging.info(
                "Round:%s | Global Acc: %.4f | Mean Local Acc: %.4f", r, global_acc, mean_trained_acc
            )

            # --- Aggregation Step ---
            if use_gcblock and fused_states:
                fused_global = fed_avg_state_dicts(fused_states, selected_weights)
                non_gc_global = fed_avg_state_dicts(non_gc_states, selected_weights)

                # Update global model
                # Note: We update the net_template, then extract state_dict to global_state_dict
                base_state = net_template.state_dict()
                for k, v in non_gc_global.items():
                    base_state[k] = v
                
                net_template.load_state_dict(base_state, strict=False)
                load_fused_weights_into_gc_model(net_template, fused_global, variant=gcfl_variant)  #GCBlock部分的权重融合
                global_state_dict = copy.deepcopy(net_template.state_dict())
            else:
                # Standard FedAvg Aggregation
                if non_gc_states: # non_gc_states holds full states if use_gcblock is False
                    global_state_dict = fed_avg_state_dicts(non_gc_states, selected_weights)

        logging.info("Homogeneous Standalone Training finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homogeneous Standalone Experiment")

    parser.add_argument("--data-name", type=str, default="fashion-mnist", choices=["cifar10", "cifar100", "mnist", "fashion-mnist"], help="dir path for dataset")
    parser.add_argument("--data-path", type=str, default="data", help="dir path for dataset")
    parser.add_argument("--num-nodes", type=int, default=100, help="number of simulated nodes")
    parser.add_argument("--fraction", type=float, default=0.1, help="number of sampled nodes in each round")

    parser.add_argument("--num-steps", type=int, default=500, help="total number of rounds")
    parser.add_argument("--optim", type=str, default="sgd", choices=["adam", "sgd"], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10, help="number of inner steps")

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

    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="Results/FedOFT_mh", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    parser.add_argument("--model", type=str, default="cnn1", choices=["cnn1", "gc_cnn1"], help="model architecture")
    parser.add_argument("--use-gcblock", type=str2bool, default=False, help="enable GCBlock fusion-aware aggregation")
    parser.add_argument("--gcfl-variant", type=str, default="A", choices=["A", "B"], help="GCBlock inflation variant")
    parser.add_argument(
        "--log-comm-to-csv", type=str2bool, default=False, help="append communication cost metrics to the output CSV"
    )

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.data_name == "cifar10":
        args.classes_per_node = 2
    elif args.data_name == "cifar100":
        args.classes_per_node = 10
    else:
        args.classes_per_node = 2

    use_gcblock = args.use_gcblock or args.model == "gc_cnn1"

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        fraction=args.fraction,
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
        seed=args.seed,
        model_name=args.model,
        use_gcblock=use_gcblock,
        gcfl_variant=args.gcfl_variant,
        log_comm_to_csv=args.log_comm_to_csv,
    )