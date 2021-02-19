# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output




        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        self.net = torch.nn.Sequential(torch.nn.Conv2d(3, 6, 5), torch.nn.LeakyReLU(), torch.nn.MaxPool2d(2, 2), torch.nn.Conv2d(6, 16, 5), torch.nn.LeakyReLU(), torch.nn.MaxPool2d(2, 2))
        self.nett = torch.nn.Sequential(torch.nn.Linear(16 * 5 * 5, 5), torch.nn.LeakyReLU(), torch.nn.Linear(5, out_size), torch.nn.Sigmoid())
        parm = list(self.net.parameters()) + list(self.nett.parameters())
        self.optimizer = torch.optim.SGD(parm, lrate)

    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = x.view(-1, 3, 32, 32)
        y = self.net(x)
        y = y.view(-1, 16 * 5 * 5)
        return self.nett(y)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        L = self.loss_fn(yhat, y)
        L.backward()
        self.optimizer.step()
        return L



def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N

    model's performance could be sensitive to the choice of learning_rate. We recommend trying different values in case
    your first choice does not seem to work well.
    """
    losses = []
    yhats = np.zeros((len(dev_set), ))
    
    nets = NeuralNet(0.01, torch.nn.CrossEntropyLoss(), train_set.shape[1], 2)
    #train_set = train_set.view((7500, 3, 32, 32))
    n = 0
    while n < (n_iter*batch_size):
        i = n % (len(train_set))
        #print(i)
        losses.append(nets.step(train_set[i:i+batch_size], train_labels[i:i+batch_size]))
        n += batch_size
    
    for i, row in enumerate(dev_set):
        yhats[i] = torch.max(nets.forward(row), 1)[1]
    return losses, yhats, nets


