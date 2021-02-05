# U-SHAPED SPLIT NEURAL NETWORK (U-SplitNN)
# Maintainer: Amrest Chinkamol (amrest.c@ku.th)
# Based on fedml_api/distributed/splitnn/server.py

import logging

import torch.nn as nn
import torch.optim as optim


class Server(object):
    """Server functional definition.

    :attr comm
    :attr model
    :attr MAX_RANK

    description: Server class is a server node working class. Providing low-level interface
    for the manager class to use.

    """

    def __init__(self, args):
        # Communication group
        self.comm = args["comm"]
        # Model
        self.model = args["model"]
        # Max rank
        self.MAX_RANK = args["max_rank"]
        self.init_params()

    def init_params(self):
        """Utility function.
        """
        self.epoch = 0
        self.log_step = 50
        self.active_node = 1
        self.train_mode()
        # TODO: Make this optimizable
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def reset_local_params(self):
        """Utility function.
        """
        self.total = 0
        # -- TO BE DELETED -- >
        # < -- END --
        self.batch_idx = 0

    def train_mode(self):
        """Set trans model to training mode.
        """
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def eval_mode(self):
        """Set trans model to evaluation mode.
        """
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def forward_pass(self, acts):
        """Forward passing the activation tensor through trans model.
        """
        self.acts = acts
        # Clean up
        self.optimizer.zero_grad()
        self.acts.retain_grad()
        # Forward pass
        self.trans_acts = self.model(acts)
        return self.trans_acts

    def backward_pass(self, grads):
        """Server backward passsing from grads tensor.
        """
        self.trans_acts.backward(grads)
        self.optimizer.step()
        return self.acts.grad

    # TODO: CHANGE THIS PROCESS
    def validation_over(self):
        """Utility function.
        This function will be called when server node received validation over signal.
        Wrapping the validation procedure and evoke training mode.
        """
        # not precise estimation of validation loss
        #self.val_loss /= self.step
        #acc = self.correct / self.total
        #logging.info("phase={} acc={} loss={} epoch={} and step={}"
        #             .format(self.phase, acc, self.val_loss, self.epoch, self.step))

        #self.epoch += 1
        #self.active_node = (self.active_node % self.MAX_RANK) + 1
        #self.train_mode()
        #logging.info("current active client is {}".format(self.active_node))
        pass
