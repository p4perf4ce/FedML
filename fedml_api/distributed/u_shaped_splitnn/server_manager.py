# U-SHAPED SPLIT NEURAL NETWORK (U-SplitNN)
# Maintainer: Amrest Chinkamol (amrest.c@ku.th)

import logging
from fedml_api.distributed.u_shaped_splitnn.message_definition import MPIMessage
from fedml_core.distributed.server.server_manager import ServerManager
from fedml_core.distributed.communication.message import Message

logger = logging.getLogger('ServerManager')
logger.setLevel(logging.DEBUG)

class USplitNNServerManager(ServerManager):
    """U-SplitNNServerManager

    This class is an abstaction class over the Server class and ServerManager Primitive class.
    """

    def __init__(self, arg_dict, trainer, backend='MPI'):
        """U-SplitNN server manager initiatiate
        """
        super().__init__(
                args=arg_dict['args'],
                comm=arg_dict['comm'],
                rank=arg_dict['rank'],
                size=arg_dict['max_rank'] + 1,
                backend=backend,
                )
        self.trainer = trainer
        self.round_idx = 0

    def run(self):
        """Starting up server process
        """
        logger.info(f"Server Initialization is complete at rank {self.trainer.rank}. Waiting for client ...")
        super().run()  # Inherit from core ServerManager class

    # --- Handlers Begin ---
    def register_message_receive_handlers(self) -> None:
        # Handle activation from smasher
        self.register_message_receive_handler(
            MPIMessage.MSG_TYPE_C2S_SEND_ACTS,
            handler_callback_func=self.handle_message_acts
        )
        # Handle gradients from header
        self.register_message_receive_handler(
                MPIMessage.MSG_TYPE_C2S_SEND_GRADS,
                handler_callback_func=self.handle_message_grads
                )
        # Handle validation mode send_validation
        self.register_message_receive_handler(
            MPIMessage.MSG_TYPE_C2S_VALIDATION_MODE,
            handler_callback_func=self.handle_message_validation_mode

        )
        # Handle validation over signal
        self.register_message_receive_handler(
                MPIMessage.MSG_TYPE_C2S_VALIDATION_OVER,
                handler_callback_func=self.handle_message_validation_over
                )
        # Handle Protocol finalization
        self.register_message_receive_handler(
                MPIMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                handler_callback_func=self.handle_message_finish_protocol
                )

    def handle_message_acts(
            self,
            msg_params,
            ) -> None:
        """Handle activation signal messages from client.
        """
        logger.info("Received activation from client")
        received_acts = msg_params.get(MPIMessage.MSG_ARG_KEY_ACTS)
        # Forward to model
        forward_acts = self.trainer.forward_pass(received_acts)
        # Forward activation signal to client
        self.send_acts_to_client(receiver_id=self.trainer.active_node, acts=forward_acts)
        logger.info(f"Forwarding propagation to Client-{self.trainer.active_node}")

    def handle_message_grads(
            self,
            msg_params,
            ) -> None:
        """Handle backward gradiant from client.
        """
        logging.info("Client gradiant feedback received...")
        # Get gradiant from client
        received_grads = msg_params.get(MPIMessage.MSG_ARG_KEY_GRADS)
        # Backward from received gradiant.
        grads = self.trainer.backward_pass(received_grads)
        # Backward activation gradiant to client
        self.send_grads_to_client(receiver_id=self.trainer.active_node, grads=grads)
        logging.info("Backwarding gradiant to client ...")

    def handle_message_validation_mode(self, msg_params)-> None:
        """Handle trigger validation signal messages from client
        """
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params) -> None:
        """Handle validation over signal message from client
        """
        self.trainer.validation_over()

    def handle_message_finish_protocol(self, msg_params) -> None:
        """Handle finalization signal from client
        """
        self.finish()

    # --- Handlers End ---

    # --- Verb Begin ---

    def send_grads_to_client(self, receiver_id, grads) -> None:
        message = Message(
                MPIMessage.MSG_TYPE_S2C_GRADS,
                sender_id=self.get_sender_id(),
                receiver_id=receiver_id
                )
        message.add_params(
                key=MPIMessage.MSG_ARG_KEY_GRADS,
                value=grads
                )
        self.send_message(message)

    def send_acts_to_client(self, receiver_id, acts) -> None:
        message = Message(
                MPIMessage.MSG_TYPE_S2C_SEND_ACTS,
                sender_id=self.get_sender_id(),
                receiver_id=receiver_id
                )
        message.add_params(
                key=MPIMessage.MSG_ARG_KEY_ACTS,
                value=acts
                )
        self.send_message(message)

    # --- Verb End ---
