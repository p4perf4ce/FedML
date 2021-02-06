import logging
from torch import nn


def model_partition(model, split_factor: float, partition_factor:float):
    """Model partitioning utility function.
    :param split_factor (0-1) spliting size of local and global model size
    :param partition_factor (0-1) smasher partition size of local model
    """
    try:
        assert split_factor > 0 and split_factor < 1
        assert partition_factor > 0 and split_factor < 1
    except AssertionError as e:
        logging.error(f"Expect open range of 0 to 1 got: {split_factor} {partition_factor}")
        raise e
    n_layers = len(nn.ModuleList(model.children()))
    # Divide Local and Global
    n_splits_local = int(n_layers * split_factor)
    n_splits_global = n_layers - n_splits_local
    # Divide smasher and header
    n_smasher = int(n_splits_local * partition_factor)
    n_header = n_splits_local - n_smasher

    # Every part must have at least one layer
    try:
        assert n_splits_local > 0
        assert n_splits_global > 0
        assert n_smasher > 0
        assert n_header > 0
    except AssertionError as e:
        logging.error(f"""Layers depth should not less than 1 got otherwise:\n
                n_splits_global: {n_splits_global}\n
                n_splits_local: {n_splits_local}\n
                n_smasher: {n_smasher}\n
                n_header: {n_header}
                """)
        raise e

    smasher = nn.Sequential(
            *nn.ModuleList(model.children())[0:n_smasher]
            )
    transfer = nn.Sequential(
            *nn.ModuleList(model.children())[n_smasher: n_layers-n_header]
            )
    header = nn.Sequential(
            *nn.ModuleList(model.children())[n_layers-n_header: ]
            )
    return smasher, transfer, header
