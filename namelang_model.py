#!/usr/bin/env python3
#    Reference(s): 
#        https://stackoverflow.com/questions/6908143/should-i-put-shebang-in-python-scripts-and-what-form-should-it-take


# namelang_model.py


# Defines a Recurrent Neural Network (RNN) to predict the language origin of a family name. 

# Requires:
#  namelang_data.py
#  namelang_train.py (to train the RNN)
#  namelang.py (to evaluate the trained RNN and perform a prediction)

# This Python script is run from within namelang_train and namelang - 
# it is not run independently.

# Adapted from the PyTorch tutorial "NLP FROM SCRATCH: CLASSIFYING NAMES 
# WITH A CHARACTER-LEVEL RNN" by Sean Robertson.
#    Reference(s):
#        https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#        https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

# The training data set is available at
#     https://download.pytorch.org/tutorial/data.zip

######################################################################

### CREATE THE RNN ###

# Before autograd, creating a recurrent neural network in Torch involved 
# cloning the parameters of a layer over several timesteps. The layers 
# held hidden state and gradients which are now entirely handled by the 
# graph itself. This means you can implement a RNN in a very “pure” way, 
# as regular feed-forward layers.
#
# This RNN module (mostly copied from the PyTorch for Torch users tutorial) 
# is just 2 linear layers which operate on an input and hidden state, 
# with a LogSoftmax layer (see below) after the output.
#
#    Reference(s): 
#        https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#        https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#example-2-recurrent-net


# [T]he softmax function ... takes as input a vector of K real numbers, 
# and normalizes it into a probability distribution consisting of K 
# probabilities proportional to the exponentials of the input numbers. 
# That is, prior to applying softmax, some vector components could be 
# negative, or greater than one, and might not sum to 1. But after applying 
# softmax, each component will be in the interval (0,1), and the components 
# will add up to 1, so that they can be interpreted as probabilities. 
# Furthermore, the larger input components will correspond to larger 
# probabilities. Softmax is often used in neural networks, to map the 
# non-normalized output of a network to a probability distribution over 
# predicted output classes.
#
#    Reference(s): 
#        https://en.wikipedia.org/wiki/Softmax_function


# Mathematically in Python:
#
#   Softmax(x) = exp(x_i) / exp(x).sum()
#   LogSoftmax(x) = log(exp(x_i) / exp(x).sum())
#
#    Reference(s):
#        https://discuss.pytorch.org/t/what-is-the-difference-between-log-softmax-and-softmax/11801
#        https://stackoverflow.com/questions/49236571/what-is-the-difference-between-softmax-and-log-softmax
#        https://datascience.stackexchange.com/questions/40714/what-is-the-advantage-of-using-log-softmax-instead-of-softmax


import torch
import torch.nn as nn
#    Reference(s):
#        https://pytorch.org/docs/stable/nn.html

from torch.autograd import Variable
# torch.autograd provides classes and functions implementing automatic 
# differentiation of arbitrary scalar valued functions. 
# [Essentially, autograd calculates the gradients between iterations of 
# an RNN. - dwh]
#    Reference(s):
#        https://pytorch.org/docs/stable/autograd.html


class RNN(nn.Module):
# Classes provide a means of bundling data and functionality together. 
# Creating a new class creates a new type of object, allowing new instances 
# of that type to be made.
#    Reference(s):
#        https://docs.python.org/3/tutorial/classes.html
#        https://www.w3schools.com/python/python_classes.asp

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

