# neuralnet.py
# Submitted by: Sooryaprakash Raja Venkataramanan
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019


import numpy as np
import torch


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """

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
        self.net = torch.nn.Sequential(torch.nn.Linear(in_size, 120), torch.nn.LeakyReLU(), torch.nn.Linear(120, 80), torch.nn.LeakyReLU(), torch.nn.Linear(80, 32), torch.nn.LeakyReLU(), torch.nn.Linear(32, out_size), torch.nn.Sigmoid())
        self.optimizer = torch.optim.SGD(self.net.parameters(), lrate)



    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        return self.net(x)

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
    """
    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    """
    losses = []
    yhats = np.zeros((len(dev_set), ))
    
    nets = NeuralNet(0.001, torch.nn.CrossEntropyLoss(), train_set.shape[1], 2)

    for n in range(n_iter):
        for i in range(0, n_iter, batch_size):
            losses.append(nets.step(train_set[i:i+batch_size], train_labels[i:i+batch_size]))
    
    for i, row in enumerate(dev_set):
        yhats[i] = torch.max(nets.forward(row), 0)[1]
    return losses, yhats, nets
