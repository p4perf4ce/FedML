# U-SHAPED SPLIT NEURAL NETWORK (U-SplitNN)
# Maintainer: Amrest Chinkamol (amrest.c@ku.th)

from fedml_api.distributed.u_shaped_splitnn.client import Client
from fedml_api.distributed.u_shaped_splitnn.message_definition import MPIMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message


import logging

class USplitNNClientManager(ClientManager):
    """Client Manager for U-SplitNN

    """

    def __init__(self, args_dict: dict, trainer: Client, backend: str = "MPI"):
        super().__init__(
                args=args_dict["args"],
                comm=args_dict["comm"],
                rank=args_dict["rank"],
                size=args_dict["size"],
                backend=backend)
        # Manager used same trainer for every client.
        self.trainer: Client = trainer
        self.trainer.train_mode()
        # Manager
        self.round_idx = 0

    def run(self):
        """Start ClientManager Training

        :description Manager must start training process from rank 1.
        """
        if self.trainer.rank == 1:
            logging.info("Starting Protocol from rank 1 process")
            # NOTE: Unusable yet, need to wait for server return.
            self.run_forward_pass()

    # --- Begin Handler Section ---       

    def register_message_receive_handlers(self) -> None:
        """Register MPI message receive handler.
        """
        # Register Client to Client Semaphore
        self.register_message_receive_handler(
            msg_type=MPIMessage.MSG_TYPE_C2C_SEMAPHORE,
            handler_callback_func=self.handle_message_semaphore)
        # Register Server to Client Gradient Update

        # TODO: Register Client to Server Gradient Update

    def handle_message_semaphore(self, msg_params) -> None:
        """C2C Semaphore callback

        :param msg_params MPI_MESSAGE parameter
        
        """
        self.trainer.train_mode()
        self.run_forward_pass()

    def handle_message_gradients(self, msg_params) -> None:
        """C2S,S2C Gradient send over.
        :param msg_params MPI_MESSAGE parameter
        """
        # Get gradient from message.
        grad = msg_params.get(MPIMessage.MSG_ARG_KEY_GRADS)
        # Begin backward pass with sender gradient.
        self.trainer.backward_pass(grads)
        # Handle when all traindata is ran out.
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            logging.info(f"Epoch over at node {self.rank}")
            self.round_idx += 1
            self.run_eval()  # Set to evaluation mode.
        else:
            self.run_forward_pass()  # Continue forward pass.

    # --- End Handler Section ---

    # --- Verb Functions Start ---
    
    def send_activations_to_server(
            self,
            acts,
            receiver_id,
            ):
        """C2S verb activation to server.
        """
        # Compose message
        message = Message(
                MPIMessage.MSG_TYPE_C2S_SEND_ACTS,
                sender_id=self.get_sender_id,
                receiver_id=receiver_id,
                )
        self.send_message(message=message)
    
    def send_semaphore_to_client(
            self,
            receiver_id,
            ):
        """C2C verb semaphore to client
        """
        # Compose message
        message = Message(
                MPIMessage.MSG_TYPE_C2C_SEMAPHORE,
                sender_id=self.get_sender_id(),
                receiver_id=receiver_id
                )
        self.send_message(message=message)

    def send_validation_signal_to_server(
            self,
            receiver_id,
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
            receiver_id,
            )
        """C2S verb validation phase ending to server
        """
        message = Message(
                MPIMessage.MSG_TYPE_C2S_VALIDATION_OVER,
                sender_id=self.get_sender_id,
                receiver_id=receiver_id
                )

    def send_finish_to_server(
            self,
            receiver_id,
            ):
        """C2S verb finalization complete signal to server
        """
        message = Message(
                MPIMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                sender_id=self.get_sender_id(),
                receiver_id=receiver_id
                )
        self.send_message(message)

    # --- Verb Function End ---
