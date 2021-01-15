# U-SHAPED SPLIT NEURAL NETWORK (U-SplitNN)
# Maintainer: Amrest Chinkamol (amrest.c@ku.th)

from multiprocessing import Semaphore
from torch.functional import Tensor
from fedml_api.distributed.u_shaped_splitnn import client
from fedml_api.distributed.u_shaped_splitnn.client import Client
from fedml_api.distributed.u_shaped_splitnn.message_definition import MPIMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message


import logging

class U_SplitNNClientManager(ClientManager):
    """Client Manager for U-SplitNN

    """

    def __init__(self, args_dict: dict, client: Client, backend: str = "MPI"):
        super().__init__(
                args=args_dict["args"],
                comm=args_dict["comm"],
                rank=args_dict["rank"],
                size=args_dict["size"],
                backend=backend)
        # Manager used same trainer for every client.
        self.client: Client = client
        # Start client as train mode
        self.client.train_mode()
        # Manager
        self.round_idx = 0

    def run(self):
        """Start ClientManager Training

        :description Manager with rank 1 must start training process.
        """
        if self.client.rank == 1:
            logging.info("Starting Protocol from rank 1 process")
            self.run_forward_pass()

    # --- Begin Handler Section ---

    def register_message_receive_handlers(self) -> None:
        """Register MPI message receive handler.
        """
        # Register Client to Client Semaphore handler
        self.register_message_receive_handler(
            msg_type=MPIMessage.MSG_TYPE_C2C_SEMAPHORE,
            handler_callback_func=self.handle_message_semaphore)
        # Register Server to Client Gradient Update handler
        self.register_message_receive_handler(
            msg_type=MPIMessage.MSG_TYPE_S2C_GRADS,
            handler_callback_func=self.handle_message_gradients)
        # Register Server to Client Activations handler
        self.register_message_receive_handler(
            msg_type=MPIMessage.MSG_TYPE_S2C_GRADS,
            handler_callback_func=self.handle_message_acts)

    def handle_message_semaphore(self, msg_params: Message) -> None:
        """C2C Semaphore callback

        :param msg_params MPI_MESSAGE parameter

        :description When this handler is called, evoking this node to training.
        """
        self.client.train_mode()
        self.run_forward_pass()

    def handle_message_gradients(self, msg_params: Message) -> None:
        """S2C Gradient send over.
        :param msg_params MPI_MESSAGE parameter
        """
        # Get gradient from message.
        grads = msg_params.get(MPIMessage.MSG_ARG_KEY_GRADS)
        # Begin backward pass with sender gradient.
        self.client.smasher_backward_pass(grads)
        # Handle when all traindata is ran out.
        if self.client.train_batch_idx == len(self.client.trainloader):
            logging.info(f"Epoch over at node {self.rank}")
            self.round_idx += 1
            self.run_validation_forward_pass()  # Run evaluation.
        else:
            self.run_forward_pass()  # Continue forward pass.

    def handle_message_acts(self, msg_params: Message) -> None:
        """S2C Activations send over.
        :param msg_params MPI_MESSAGE parameter
        """
        # Passing activation to client
        acts = msg_params.get(MPIMessage.MSG_ARG_KEY_ACTS)
        loss = self.client.header_forward_pass(trans_acts=acts)
        # Handle train/test
        if self.client == 'train':
            header_grads = self.client.header_backward_pass()
            self.send_gradients_to_server(grads=header_grads, receiver_id=self.client.SERVER_RANK)
        elif self.client == 'validation':
            # NOTE: Add report script here.
            # Proceed to next batch
            if self.client.validate_batch_idx < len(self.client.testloader):
                self.client.smasher_forward_pass()
                self.client.validate_batch_idx += 1
                return;
            else:
                # End validation phase
                self.round_idx += 1
                self.send_validation_over_to_server(receiver_id=self.client.SERVER_RANK)
                if self.round_idx == self.client.MAX_EPOCH_PER_NODE:
                    if self.client.rank == self.client.MAX_RANK:
                        # Send gratituous to server.
                        self.send_finish_to_server(receiver_id=self.client.SERVER_RANK)
                    self.finish()
                elif self.round_idx < self.client.MAX_EPOCH_PER_NODE:
                    logging.info(f'Sending Semaphore from {self.client.rank} to {self.client.node_right}')
                    self.send_semaphore_to_client(receiver_id=self.client.node_right)


    # --- End Handler Section ---

    # --- Verb Functions Start ---

    def send_gradients_to_server(
            self,
            grads: Tensor,
            receiver_id: int,
            ):
        """C2S verb gradient to server.
        """
        message = Message(
                MPIMessage.MSG_TYPE_C2S_SEND_GRADS,
                sender_id=self.get_sender_id,
                receiver_id=receiver_id,
                )
        message.add_params(
                MPIMessage.MSG_ARG_KEY_GRADS,
                value=grads
                )
        self.send_message(message=message)

    def send_activations_to_server(
            self,
            acts: Tensor,
            receiver_id: int,
            ):
        """C2S verb activation to server.
        """
        # Compose message
        message = Message(
                MPIMessage.MSG_TYPE_C2S_SEND_ACTS,
                sender_id=self.get_sender_id,
                receiver_id=receiver_id,
                )
        message.add_params(
                key=MPIMessage.MSG_ARG_KEY_ACTS,
                value=acts,
                )
        self.send_message(message=message)

    def send_semaphore_to_client(
            self,
            receiver_id: int,
            ):
        """C2C verb semaphore to client
        """
        # Compose message
        message = Message(
                MPIMessage.MSG_TYPE_C2C_SEMAPHORE,
                sender_id=self.get_sender_id(),
                receiver_id=receiver_id)
        self.send_message(message=message)

    def send_validation_signal_to_server(
            self,
            receiver_id: int,
            ):
        """C2S verb semaphore to server
        """
        # Compose message
        message = Message(
                MPIMessage.MSG_TYPE_C2S_VALIDATION_MODE,
                sender_id=self.get_sender_id(),
                receiver_id=receiver_id
                )
        self.send_message(message=message)

    def send_validation_over_to_server(
            self,
            receiver_id: int,
            ):
        """C2S verb validation phase ending to server
        """
        message = Message(
                MPIMessage.MSG_TYPE_C2S_VALIDATION_OVER,
                sender_id=self.get_sender_id,
                receiver_id=receiver_id
                )
        self.send_message(message=message)

    def send_finish_to_server(
            self,
            receiver_id: int,
            ):
        """C2S verb finalization complete signal to server
        """
        message = Message(
                MPIMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                sender_id=self.get_sender_id(),
                receiver_id=receiver_id
                )
        self.send_message(message=message)

    # --- Verb Function End --

    # --- Functional Begin ---

    def run_forward_pass(self):
        """Run client forward pass function.
        """
        # Run forward pass
        smashed_acts = self.client.smasher_forward_pass()
        # Send to server
        self.send_activations_to_server(
                acts=smashed_acts,
                receiver_id=self.client.SERVER_RANK)

        self.client.train_batch_idx += 1

    def run_validation_forward_pass(self):
        """Run validation forward pass.
        """
        # Initiate validation
        self.send_validation_signal_to_server(receiver_id=self.client.SERVER_RANK)
        # Prep local
        self.client.eval_mode()
        # Execute validation
        self.client.smasher_forward_pass()
        self.client.train_batch_idx += 1

# ---   Function End   ---

