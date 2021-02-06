# Template from split_nn
from types import SimpleNamespace
import click
import logging
import os
import sys
import traceback

import numpy as np
import setproctitle
import torch
import torch.nn as nn
from mpi4py import MPI
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file 
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_distributed_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_distributed_cinic10
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_distributed_cifar10

from fedml_api.distributed.u_shaped_splitnn.u_shaped_splitnn_api import Initialize, UShapedSplitNN_Initialize
from fedml_api.distributed.u_shaped_splitnn.utils import model_partition
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56

from fedml_experiments.distributed.u_shaped_splitnn.base_cli import add_options, simple_options, stat_rep


@click.group()
def cli():
    """U-Shaped Split Neural Network experiment helper interface
    """
    pass


@cli.command()
@add_options(simple_options)
def run(**kwargs):
    """
    Running an experiment
    """
    comm, process_id, worker_number = Initialize()
    args = SimpleNamespace(**kwargs)
    click.clear()
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    if process_id == 0:
        logger = logging.getLogger('cli-interface')
        logger.setLevel(loglevel)
    else:
        logger = logging.getLogger('Slave-{process_id}')
        loglevel = logging.WARNING
        logger.setLevel(logging.WARNING)
    logger.propagate = False
    logfmt = str(process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
    datefmt = '%a, %d %b %Y %H:%M:%S'
    fmt = logging.Formatter(fmt=logfmt, datefmt=datefmt)

    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(loglevel)
    stdhandler.setFormatter(fmt)
    logger.addHandler(stdhandler)

    logger.info(f"Logging started.")
    logger.debug(f"Logging in VERBOSE mode.")

    click.echo(stat_rep(kwargs))
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)

    str_process_name = "U-Shaped SplitNN (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)
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

    # Partition the data according to process_id
    train_data_num, train_data_global, \
    test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = data_loader(process_id, args.dataset, args.data_dir,
                                                               args.partition_method, args.partition_alpha,
                                                               args.client_number, args.batch_size)

    # create the model
    model = None
    split_factor = 0.3
    partition_factor = 0.5
    if args.model == "mobilenet":
        model = mobilenet(class_num=class_num)
    elif args.model == "resnet56":
        model = resnet56(class_num=class_num)

    n_layers = len(nn.ModuleList(model.children()))
    # partition_indice = int(n_layers*split_factor*partition_factor)

    fc_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Flatten(),
                             nn.Linear(fc_features, class_num))
    logger.info("Splitting the model ...")
    # Split The model
    # smasher_model = nn.Sequential(*nn.ModuleList(model.children())[0:partition_indice])
    # header_model = nn.Sequential(*nn.ModuleList(model.children())[n_layers - partition_indice:])
    # client_model = [smasher_model, header_model]
    # server_model = nn.Sequential(*nn.ModuleList(model.children())[partition_indice: n_layers - partition_indice])
    smasher_model, server_model, header_model = model_partition(model=model, partition_factor=partition_factor, split_factor=split_factor)
    client_model = [smasher_model, header_model]
    logging.info("Splitting process completed ...")
    #logging.info("Model Partition:")
    #logging.info(f"{print(smasher_model)}")
    #logging.info(f"{print(server_model)}")
    #logging.info(f"{print(header_model)}")

    logger.info("Initializing ...")
    try:
        UShapedSplitNN_Initialize(process_id, worker_number, device, comm,
                            client_model, server_model, train_data_num,
                            train_data_global, test_data_global, local_data_num,
                            train_data_local, test_data_local, args)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt Caughted.")
        if process_id != 0:
            MPI.COMM_WORLD.Abort()
            logger.info("Broadcast abort signal to entire communication group...")
        elif click.confirm("Ending an experiment ?"):
            MPI.COMM_WORLD.Abort()
            logger.info("Broadcast abort signal to entire communication group...")
            sys.exit(1)
        else:
            logger.info("Resuming...")

    except Exception as e:
        logger.critical("Critical error caught while running an experiment. ABORTING.")
        logger.critical(f"ErrorType: {type(e).__name__}")
        logger.critical(f"Info: {traceback.format_exc()}")
        MPI.COMM_WORLD.Abort()
        logger.info("Broadcast abort signal to entire communication group...")
        raise e
    logger.info("Experiment has ended.")

if __name__ == '__main__':
    cli()
