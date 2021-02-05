# U-shaped SplitNN definition
# Based on fedml_api/distributed/splitnn/message_define.py

class MPIMessage(object):
    """MPI Message definition class.
    Server-to-Client
    :attr MSG_TYPE_S2C_GRADS                Server gradient to Client message.
    :attr MSG_TYPE_S2C_ACTS                 Server activations to Client message.

    Client-to-Server
    :attr MSG_TYPE_C2S_SEND_ACTS            Client activations to Server message.
    :attr MSG_TYPE_C2S_VALIDATION_MODE      Client validation trigger signal to Server message.
    :attr MSG_TYPE_C2S_VALIDATION_OVER      Client validation over signal to Server message.
    :attr MSG_TYPE_C2S_PROTOCOL_FINISHED    Client finalization signal to Server message.

    Client-to-Client
    :attr MSG_TYPE_C2C_SEMAPHORE            Client semaphore to Client message.

    description: This class is a MPI_MESSAGE definition class.
    """
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_SEND_ACTS = 2

    # Client to server
    MSG_TYPE_C2S_SEND_ACTS = 3
    MSG_TYPE_C2S_SEND_GRADS = 4
    MSG_TYPE_C2S_VALIDATION_MODE = 5
    MSG_TYPE_C2S_VALIDATION_OVER = 6
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 7

    # Client to client
    MSG_TYPE_C2C_SEMAPHORE = 8  # Interlocking

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
    Message payload keywords definition.
    """
    MSG_ARG_KEY_ACTS = "activations"
    MSG_ARG_KEY_GRADS = "activation_grads"
