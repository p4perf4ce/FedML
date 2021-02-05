# Template from split_nn
from types import SimpleNamespace
import click
import logging
import os
import sys

import numpy as np
import setproctitle
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_distributed_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_distributed_cinic10
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_distributed_cifar10

from fedml_api.distributed.u_shaped_splitnn.u_shaped_splitnn_api import Initialize, UShapedSplitNN_Initialize
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56

from fedml_experiments.distributed.u_shaped_splitnn.base_cli import add_options, simple_options, stat_rep

def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


@click.command()
@add_options(simple_options)
def run(**kwargs):
    click.clear()
    click.echo(stat_rep(kwargs))
    click.confirm('Continue to run with this settings ?', abort=True)
    args = SimpleNamespace(**kwargs)
    

    comm, process_id, worker_number = Initialize()

    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)

    str_process_name = "U-Shaped SplitNN (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(worker_number)

    # load data
    if args.dataset == "cifar10":
        data_loader = load_partition_data_distributed_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_distributed_cifar100
    elif args.dataset == "cinic10":
        data_loader = load_partition_data_distributed_cinic10
    else:
        data_loader = load_partition_data_distributed_cifar10

    train_data_num, train_data_global, \
    test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = data_loader(process_id, args.dataset, args.data_dir,
                                                               args.partition_method, args.partition_alpha,
                                                               args.client_number, args.batch_size)

    # create the model
    model = None
    split_factor = 0.5
    partition_factor = 0.5
    if args.model == "mobilenet":
        model = mobilenet(class_num=class_num)
    elif args.model == "resnet56":
        model = resnet56(class_num=class_num)

    n_layers = len(nn.ModuleList(model.children()))
    partition_indice = int(n_layers*split_factor*partition_factor)

    fc_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Flatten(),
                             nn.Linear(fc_features, class_num))
    logging.info("Splitting the model ...")
    # Split The model
    smasher_model = nn.Sequential(*nn.ModuleList(model.children())[0:partition_indice])
    header_model = nn.Sequential(*nn.ModuleList(model.children())[n_layers - partition_indice:])
    client_model = [smasher_model, header_model]
    server_model = nn.Sequential(*nn.ModuleList(model.children())[partition_indice: n_layers - partition_indice])
    logging.info("Splitting process completed ...")

    logging.info("Initializing ...")
    UShapedSplitNN_Initialize(process_id, worker_number, device, comm,
                        client_model, server_model, train_data_num,
                        train_data_global, test_data_global, local_data_num,
                        train_data_local, test_data_local, args)

if __name__ == '__main__':
    run()
