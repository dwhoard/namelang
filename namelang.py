#!/usr/bin/env python3
#    Reference(s): 
#        https://stackoverflow.com/questions/6908143/should-i-put-shebang-in-python-scripts-and-what-form-should-it-take


# namelang.py


# Uses a previously trained Recurrent Neural Network (RNN; see 
# namelang_train.py) to predict the language origin of a family name 
# specified on the command line.

# Requires:
#  namelang_data.py
#  namelang_model.py
#  namelang_train.py (to train the RNN)

# Example:
#
# % python namelang.py Schmidt
# 
# Language Origin (Probability):
#   1. German (0.38)
#   2. English (0.20)
#   3. French (0.15)

# Adapted from the PyTorch tutorial "NLP FROM SCRATCH: CLASSIFYING NAMES 
# WITH A CHARACTER-LEVEL RNN" by Sean Robertson.
#    Reference(s):
#        https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#        https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

# The training data set is available at
#     https://download.pytorch.org/tutorial/data.zip

######################################################################

# Before doing anything else, check for improper command line argument(s)
import sys

if len(sys.argv) < 2:
    print("***ERROR: Please supply a name to parse on the command line")
    exit()
else:
    insample=sys.argv[1]

if len(sys.argv) > 2:
    print("***ERROR: too many names on the command line; using the first one only")


# Now start the main routine
import torch
import torch.nn as nn

from numpy import exp

from namelang_data import *
from namelang_model import *


# Function to evaluate the pre-trained RNN model
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


### RESTORE THE TRAINED RNN ###

# There are two main approaches for serializing and restoring a model. 
# The first (recommended) saves and loads only the model parameters:
#
#    torch.save(the_model.state_dict(), PATH)
#
# Then later:
#
#    the_model = TheModelClass(*args, **kwargs)
#    the_model.load_state_dict(torch.load(PATH))
#
# The second saves and loads the entire model:
#
#    torch.save(the_model, PATH)
#
# Then later:
#
#    the_model = torch.load(PATH)
#
# In the second case, the serialized data is bound to the specific classes 
# and the exact directory structure used, so it can break in various ways 
# when used in other projects, or after some serious refactors.
#
# Nonetheless, we are using the second method here.
#    Reference(s):
#        https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch


#rnn = torch.load("namelang_rnn_trained.pth")
rnn = torch.load("namelang_rnn_trained_500K.pth")


#####################################################################
### INTERLUDE: Backpropagation (backprop)

# "In machine learning, specifically deep learning, backpropagation is 
# an algorithm widely used in the training of feedforward neural networks 
# for supervised learning; generalizations exist for other artificial 
# neural networks, and for functions generally. Backpropagation efficiently 
# computes the gradient of the loss function with respect to the weights 
# of the network for a single input-output example. This makes it feasible 
# to use gradient methods for training multi-layer networks, updating 
# weights to minimize loss; commonly one uses gradient descent or variants 
# such as stochastic gradient descent. The backpropagation algorithm works 
# by computing the gradient of the loss function with respect to each 
# weight by the chain rule, iterating backwards one layer at a time from 
# the last layer to avoid redundant calculations of intermediate terms 
# in the chain rule..." 
#    Reference(s):
#        "Backpropagation", Wikipedia - https://en.wikipedia.org/wiki/Backpropagation

# "The term back-propagation is often misunderstood as meaning the whole 
# learning algorithm for multi layer neural networks. Actually, 
# back-propagation refers only to the method for computing the gradient, 
# while another algorithm,such as stochastic gradient descent, is used 
# to perform learning using this gradient."
#    Reference(s):
#        Goodfellow, I., Bengio, Y., & Courville, A., 2016, Deep Learning, 
#            MIT Press, p. 200 - "Back-Propagation and Other Differentiation Algorithms" (ISBN 9780262035613)
#####################################################################


def predict(insample, n_predictions=3):
    #print('\n> %s' % insample)
    with torch.no_grad():
    # torch.no_grad() impacts the autograd engine and deactivates it.
    # It will reduce memory usage and speed up computations but you 
    # won’t be able to backprop (which you don’t want in an eval script).
    #    Reference(s):
    #        https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615

        output = evaluate(line2tensor(unicode2ascii(insample)))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        # Print results
        print('')
        print('Language Origin (Probability):')
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('  %i. %s (%.2f)' % (i+1, all_categories[category_index], exp(value)))
            # When the model is trained with nn.LogSoftmax + nn.NLLLoss, 
            # exp(output) gives the probability of the prediction being correct.
            #    Reference(s):
            #        https://discuss.pytorch.org/t/trouble-getting-probability-from-softmax/26764/4 
            #        https://discuss.pytorch.org/t/understanding-nllloss-function/23702

            predictions.append([value, all_categories[category_index]])
        print('')

predict(insample)

