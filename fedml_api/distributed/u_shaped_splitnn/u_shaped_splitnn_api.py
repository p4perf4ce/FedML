from mpi4py import MPI

from fedml_api.distributed.u_shaped_splitnn.client import Client
from fedml_api.distributed.u_shaped_splitnn.client_manager import USplitNNClientManager
from fedml_api.distributed.u_shaped_splitnn.server import Server
from fedml_api.distributed.u_shaped_splitnn.server_manager import USplitNNServerManager


def Initialize():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number

def init_server(comm, server_model, process_id, worker_number, device, args):
    arg_dict = {"comm": comm, "model": server_model, "max_rank": worker_number - 1,
                "rank": process_id, "device": device, "args": args}
    server = Server(arg_dict)
    server_manager = USplitNNServerManager(arg_dict, server)
    server_manager.run()

def init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                process_id, server_rank, epochs, device, args):
    arg_dict = {"comm": comm, "trainloader": train_data_local, "testloader": test_data_local,
                "model": client_model, "rank": process_id, "server_rank": server_rank,
                "max_rank": worker_number - 1, "epochs": epochs, "device": device, "args": args}
    client = Client(arg_dict)
    client_manager = USplitNNClientManager(arg_dict, client)
    client_manager.run()

def UShapedSplitNN_Initialize(process_id: int, worker_number: int, device, comm, client_model,
                        server_model, train_data_num, train_data_global, test_data_global,
                        local_data_num, train_data_local, test_data_local, args):
    server_rank = 0
    if process_id == server_rank:
        # Initialize server as it is.
        init_server(comm, server_model, process_id, worker_number, device, args)
    else:
        # Initialize client as it is
        init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                    process_id, server_rank, args.epochs, device, args)
    pass

