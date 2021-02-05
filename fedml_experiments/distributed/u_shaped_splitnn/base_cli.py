import click
from types import SimpleNamespace



simple_options = [
    click.option('--model', type=str, default='resnet56', metavar='N',
                           help='neural network used in training'),
    click.option('--dataset', type=str, default='cifar10', metavar='N',
                           help='dataset used for training'),
    click.option('--data_dir', type=str, default='./../../../data/cifar10',
           help='data directory'),
    click.option('--partition_method', type=str, default='hetero', metavar='N',
           help='how to partition the dataset on local workers'),
    click.option('--partition_alpha', type=float, default=0.5, metavar='PA',
           help='partition alpha (default: 0.5)'),
    click.option('--client_number', type=int, default=16, metavar='NN',
           help='number of workers in a distributed cluster'),
    click.option('--batch_size', type=int, default=64, metavar='N',
           help='input batch size for training (default: 64)'),
    click.option('--lr', type=float, default=0.001, metavar='LR',
           help='learning rate (default: 0.001)'),
    click.option('--wd', type=float, default=0.001,
           help='weight decay parameter'),
    click.option('--epochs', type=int, default=5, metavar='EP',
           help='how many epochs will be trained locally'),
    click.option('--local_points', type=int, default=5000, metavar='LP',
           help='the approximate fixed number of data points we will have on each local worker'),
    click.option('--comm_round', type=int, default=10,
           help='how many round of communications we shoud use'),
    click.option('--frequency_of_the_test', type=int, default=1,
           help='the frequency of the algorithms'),
    click.option('--gpu_server_num', type=int, default=1,
           help='gpu_server_num'),
    click.option('--gpu_num_per_server', type=int, default=4,
           help='gpu_num_per_server'),
    click.option('--verbose', is_flag=True, default=False),
]

def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options

def stat_rep(options):
    header = "--- --- --- Options --- --- ---\n"
    stat=''.join([ f"{opt}: {val}\n" for opt, val in options.items()])
    footer = "--- --- --- Options --- --- ---\n"
    return header + stat + footer

@click.group()
def base():
    """Example cli
    """
    pass

@base.command()
@add_options(simple_options)
def run(
        model:str, dataset:str, data_dir:str, partition_method: str, partition_alpha: str,
        client_number: int, batch_size:int, lr: float, wd: float, epochs: float,
        local_points: int, comm_round: int, frequency_of_the_test: int, gpu_server_num: int,
        gpu_num_per_server: int, verbose: bool):
    """ This command will run the entire experiment.
    """
    click.echo(stat_rep(locals()))
    pass

@base.command()
@add_options(simple_options)
def simple_run(**kwargs):
    """This command avoided manually typing all options, parameter amd flags
    """
    args = SimpleNamespace(**kwargs)


if __name__ == '__main__':
    print("Base Call [DEBUGGING]")
    base()
