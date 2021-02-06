# U-SHAPED SPLIT NEURAL NETWORK (U-SplitNN)
# Maintainer: Amrest Chinkamol (amrest.c@ku.th)
import logging
from typing import Any, Collection, Iterator, Tuple

from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim

from typing_extensions import Literal


class Client(object):
    """Client worker on node.

    :attr comm          MPI_COM_GROUP
    :attr model         SplitNN model(s)
    :attr trainloader   Iterable trainloader object
    :attr testloader    Iterable testloader object
    :attr dataloader    Iterable dataloader object
    :attr rank          Client's MPI_NODE_RANK
    :attr MAX_RANK      Client Communation group MPI_MAX_RANK
    :attr node_left     Before client node rank
    :attr node_right    Next client node rank
    :attr epoch_count   Epoch counter
    :attr server_rank   Server's MPI_NODE_RANK
    :attr optimizer     Model optimizer(s)
    :attr inputs        Current input data

    description: This class is client worker process class, serve to mimicking the client node in the system.
    Client should be able to forward through `smasher` and send result tensor to server node and then receive
    tensor from server node to forward through `header` to obtain final result.

    This class handle the lowest level of work that need to be done, in order to provide the manager an abstract function.

    There shouldn't be any need to override this class, (We should move model manipulation things out into other abstracts class)
    """

    def __init__(self, args):
        # MPI Communation group
        self.comm = args["comm"]
        self.smasher = args["model"][0]
        self.header = args["model"][1]
        # Dataloader
        self.trainloader: Collection[Any] = args["trainloader"]
        self.testloader: Collection[Any] = args["testloader"]
        self.dataloader: Iterator[tuple]
        # MPI Node rank
        self.rank: int = args["rank"]
        # MPI Communation group MAX_RANK
        self.MAX_RANK: int = args["max_rank"]
        # Find neighbor node
        # NOTE: Ring Topology.
        self.node_left: int = self.MAX_RANK if self.rank == 1 else self.rank - 1
        self.node_right: int = 1 if self.rank == self.MAX_RANK else self.rank + 1
        # Settings
        self.MAX_EPOCH_PER_NODE: int = args["epochs"]
        self.SERVER_RANK: int = args["server_rank"]
        # TODO: This should be optimizable ?
        # NOTE: <PROPOSE> This should come from the model entirely
        # For easy slicing model. This functionality should retain ?
        self.smasher_optimizer: optim.Optimizer = optim.SGD(self.smasher.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=5e-4)
        self.header_optimizer: optim.Optimizer = optim.SGD(self.header.parameters(), lr=0.1, momentum=0.9,
                                    weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()  # TODO: Configurable
        self.device = args["device"]
        # Data handling
        self._labels: Tensor
        # Local parameters
        self._total: int = 0
        self._correct: int = 0
        self._val_loss: int = 0
        self._epoch_count: int = 0
        self.train_batch_idx: int = 0
        self.validate_batch_idx: int = 0
        self.phase: Literal['validation', 'train']
        self._log_step: int = 50
        self._step: int = 0

    def reset_local_params(self):
        self._total = 0
        self._val_loss = 0
        self._step = 0
        self.train_batch_idx = 0
        self.validate_batch_idx = 0

    def eval_mode(self) -> None:
        """Set smasher and transcriber into evaluation mode.
        dataloader attribute will load the test data.
        """
        self.dataloader = iter(self.testloader)
        self.phase = "validation"
        self.smasher.eval()
        self.header.eval()

    def train_mode(self) -> None:
        """Set smasher and transcriber into training mode.
        dataloader attribute will load the train data.
        """
        self.dataloader = iter(self.trainloader)
        self.phase = "train"
        self.smasher.train()
        self.header.train()


    # NOTE: We should devide forward/backward passing
    #       process into two separated section.
    # NOTE: DONE


    def smasher_forward_pass(self) -> Tensor:
        """Forward pass of U-shaped SplitNN

        description: Forward pass of U-shaped SplitNN contain 3 phases
        smashing phase, transfering phase, transcribe phase.
        Where smashing phase and transcribe phase belong to the client
        and transfering phase belong to the server.
        """

        # Get local data up.
        inputs, labels = next(self.dataloader)
        # Transfer tensor to device.
        inputs, self._labels = inputs.to(self.device), labels.to(self.device)
        # Set smasher gradient to zero.
        self.smasher_optimizer.zero_grad()
        # Passing to smasher part.
        self.smasher_acts: Tensor = self.smasher(inputs)

        return self.smasher_acts

    def _accuracy(self, logits: Tensor, labels: Tensor) -> Any:
        """Accuracy calculation helper function.

        :description This function serve the calculation of the accuracy,
                     and should always be overidable.

        :return Accuracy in train mode and None in validation mode.
        """
        self._total += labels.size(0)
        prediction: Tuple[Any, Tensor] = logits.max(1)
        _, idx = prediction
        self._correct += idx.eq(labels).sum().item()
        if self._step % self._log_step == 0 and self.phase == 'train':
            accuracy = self._correct / self._total
            logging.info(f'phase=train acc={accuracy} loss={self.loss} epoch={self._epoch_count} step={self._step}'
                    )
            return accuracy
        else:
            self._val_loss += self.loss.item()


    def header_forward_pass(self, trans_acts: Tensor) -> None:
        """Forward pass of U-shaped SplitNN on header part
        """
        self.transfer_acts = trans_acts
        self.header_optimizer.zero_grad()
        self.transfer_acts.retain_grad()
        # Header Passing
        logits = self.header(trans_acts)

        self.loss: Tensor = self.criterion(logits, self._labels)
        # Calculate Accuracy
        self._accuracy(logits=logits, labels=self._labels)
        self._step += 1
        return self.loss


    def header_backward_pass(self) -> Tensor:
        """Backward pass of U-shaped SplitNN

        description: Backward gradient from transcriber part to server and backaround
        """
        self.loss.backward()
        self.header_optimizer.step()
        return self.transfer_acts.grad

    def smasher_backward_pass(self, trans_grads: Tensor) -> None:
        """Backward pass of U-shaped SplitNN
        description: Backward gradient from transfer part on server back to smasher.
        """
        # Received gradient from server
        self.smasher_acts.backward(trans_grads)
        self.smasher_optimizer.step()

