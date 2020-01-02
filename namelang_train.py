#!/usr/bin/env python3
#    Reference(s): 
#        https://stackoverflow.com/questions/6908143/should-i-put-shebang-in-python-scripts-and-what-form-should-it-take


# namelang_train.py


# Trains a Recurrent Neural Network (RNN) to predict the language origin 
# of a family name. 

# Requires:
#  namelang_data.py
#  namelang_model.py
#  namelang.py (to evaluate the trained RNN and perform a prediction)

# Example:
#
# % python namelang_train

# Adapted from the PyTorch tutorial "NLP FROM SCRATCH: CLASSIFYING NAMES 
# WITH A CHARACTER-LEVEL RNN" by Sean Robertson.
#    Reference(s):
#        https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#        https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

# The training data set is available at
#     https://download.pytorch.org/tutorial/data.zip

######################################################################

import torch
import torch.nn as nn

from namelang_data import *
from namelang_model import *

import random
import time
import math


### PRINT EXTRA DIAGNOSTIC INFO AS THE ROUTINE RUNS?
#VERBOSE = True
VERBOSE = False


### DEFINE THE NUMBER OF ITERATIONS (AND VISUALIZATION SAMPLING)
#n_iters = 1000    
#print_every = 50
#plot_every = 10

#n_iters = 10000
#print_every = 500
#plot_every = 100

#n_iters = 100000
#print_every = 5000
#plot_every = 1000

n_iters = 500000
print_every = 25000
plot_every = 5000


### SET UP THE RNN
n_hidden = 128   # number of hidden layers

# learning_rate is multiplied with the gradient of the loss function to 
# update the weights for backpropagation of the RNN.
# too high = the RNN might explode; too low = the RNN might not learn
learning_rate = 0.005 


#####################################################################
### INTERLUDE: What is a Recurrent Neural Network?

# Training a typical neural network involves the following steps:
#
# 1. Input an example from a dataset.
# 2. The network will take that example and apply some complex computations 
#    to it using randomly initialised variables (called weights and biases).
# 3. A predicted result will be produced.
# 4. Comparing that result to the expected value will give us an error.
# 5. Propagating the error back through the same path will adjust the 
#    variables.
# 6. Steps 1-5 are repeated until we are confident to say that our
#    variables are well-defined.
# 7. A predication is made by applying these variables to a new unseen 
#    input.
#
# Recurrent neural networks work similarly but, in order to get a clear 
# understanding of the difference, we will go through the simplest model 
# using the task of predicting the next word in a sequence based on the 
# previous ones.
#
# First, we need to train the network using a large dataset. For the 
# purpose, we can choose any large text ("War and Peace" by Leo Tolstoy 
# is a good choice). When done training, we can input the sentence "Napoleon 
# was the Emperor of ..." and expect a reasonable prediction based on the 
# knowledge from the book.
#
# So, how do we start? As explained above, we input one example at a 
# time and produce one result, both of which are single words. The difference 
# with a feedforward network comes in the fact that we also need to be 
# informed about the previous inputs before evaluating the result. So 
# you can view RNNs as multiple feedforward neural networks, passing 
# information from one to the other.
#
# Since plain text cannot be used in a neural network, we need to encode 
# the words into vectors [for example, as] one-hot encoded vectors. These 
# are (V,1) vectors (V is the number of words in our vocabulary) where 
# all the values are 0, except the one at the i-th position. For example, 
# if our vocabulary is apple, apricot, banana, ..., king, ..., zebra 
# and the word is banana, then the vector is [0, 0, 1, ..., 0, ..., 0].
#
# [Three equations - described only qualitatively here - are used for training:]
#
# 1. h_t holds information about the previous words in the sequence. 
#    It is calculated using the previous h_(t-1) vector and the current 
#    word vector x_t. We also apply a non-linear activation function f 
#    (usually tanh or sigmoid) to the final summation. It is acceptable 
#    to assume that h_0 is a vector of zeros.
# 2. y_t calculates the predicted word vector at a given time step t. 
#    We use the softmax function to produce a (V,1) vector with all elements 
#    summing up to 1. This probability distribution gives us the index 
#    of the most likely next word from the vocabulary.
# 3. J uses the cross-entropy loss function at each time step t to calculate 
#    the error between the predicted and actual word.
#
# [Equations 1 and 2 contain weight terms (W), which are matrices initialised 
# with random elements, adjusted via backpropagation using the error 
# from equation 3, the loss function.]
#
# Once we have obtained the correct weights, predicting the next word 
# in the sentence "Napoleon was the Emperor of ..." is quite straightforward. 
# Plugging each word at a different time step of the RNN would produce 
# h_1, h_2, h_3, h_4. We can derive y_5 using h_4 and x_5 (vector of 
# the word "of"). If our training was successful, we should expect 
# that the index of the largest number in y_5 is the same as the index 
# of the word "France" in our vocabulary.
#
# Problems with a standard RNN
#
# Unfortunately, if you implement the above steps, you won't be so 
# delighted with the results. That is because the simplest RNN model 
# has a major drawback, called vanishing gradient problem, which prevents 
# it from being accurate.
#
# In a nutshell, the problem comes from the fact that at each time step 
# during training we are using the same weights to calculate y_t. That 
# multiplication is also done during back-propagation. The further we 
# move backwards, the bigger or smaller our error signal becomes. This 
# means that the network experiences difficulty in memorising words from 
# far away in the sequence and makes predictions based on only the most 
# recent ones.
#
# That is why more powerful models like LSTM (Long Short Term Memory) 
# and GRU (gated Recurrent Unit) come in hand. Solving the above issue, 
# they have become the accepted way of implementing recurrent neural 
# networks.
#
#    Reference(s):
#        https://towardsdatascience.com/learn-how-recurrent-neural-networks-work-84e975feaaf7
#        LSTM: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#        GRU:  https://en.wikipedia.org/wiki/Gated_recurrent_unit

#####################################################################


### TRAINING THE RNN ###
# Function to determine maximum probability in the output tensor and list 
# the corresponding category name
def category_from_output(output):
    top_n, top_i = output.topk(1)  # topk(n) finds the top n elements in a list
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


# Function to select a random element of a list
def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


# Function to select a random training sample (a language and name pair)
def random_training_sample():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line2tensor(line)
    return category, line, category_tensor, line_tensor


# Now all it takes to train this network is show it a bunch of examples, 
# have it make guesses, and tell it if it's wrong.
#
# The loss function is a representation of how far off the RNN model's 
# output (prediction) is from the correct answer (ground-truth). For the 
# loss function, nn.NLLLoss is appropriate, since the last layer of the 
# RNN is nn.LogSoftmax.
#
# Each loop of training will:
#
# - Create input and target tensors
# - Create a zeroed initial hidden state
# - Read each letter in and
# - Keep hidden state for next letter
# - Compare final output to target
# - Back-propagate
# - Return the output and loss
#
#    Reference(s):
#        https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# To run a step of this network we need to pass an input (in our case, 
# the tensor for the current letter) and a previous hidden state (which 
# we initialize as zeros at first). We'll get back the output (probability 
# of each language) and a next hidden state (which we keep for the next 
# step). To isolate letters, slices from the output of the letter2tensor 
# function are used. This could be further optimized by pre-computing 
# batches of tensors.
#
#    Reference(s):
#        https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

input = line2tensor('Albert')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)

if VERBOSE == True:
    print('Sample input: Albert')
    print('Sample output:',output)
    print('Every item in the output tensor is the likelihood of that category.\n')

    print(category_from_output(output),'\n')

    print('Randomly selected category:',random_choice(all_categories),'\n')

# Example: randomly pick 10 training samples
if VERBOSE == True:
    print('Randomly selected training samples')
    for i in range(10):
        category, line, category_tensor, line_tensor = random_training_sample()
        print('category =', category, '/ line =', line)
    print('')

# Function to train the RNN
def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()
    return output, loss.item()

# Now we just have to run that with a bunch of examples. Since the train 
# function returns both the output and loss we can print its guesses 
# and also keep track of loss for plotting. Since there are 1000s of 
# examples we print only every print_every examples, and take an average 
# of the loss.
#
#    Reference(s):
#        https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Keep track of losses for plotting
current_loss = 0
all_losses = []

# Function to track runtime
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

print('Training the RNN:')

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_sample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


### VISUALIZING AND EVALUATING THE RESULTS ###
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Define font sizes
ptextbig = 20
ptextmedium = 16
ptextsmall = 12


### FIRST PLOT - LOSSES
# Plotting the historical loss from all_losses shows the network learning

# The smaller the loss between iterations, the better the RNN has learned - dwh

# Set up plot
fig = plt.figure(figsize=(9.0,7.5))
ax = fig.add_subplot(111)
plt.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.95)

ax.plot(all_losses)

plt.title("RNN Losses (N="+str(n_iters)+")", fontsize=ptextbig, pad=10)
plt.xlabel("Iteration / "+str(plot_every), fontsize=ptextmedium, labelpad=10)
plt.ylabel("Loss", fontsize=ptextmedium, labelpad=10)

plt.minorticks_on()

ax.tick_params(axis='both', which='major', labelsize=ptextsmall)

plt.show(block=False)


### SECOND PLOT - CONFUSION
# To see how well the network performs on different categories, we will 
# create a confusion matrix, indicating for every actual language (rows) 
# which language the network guesses (columns). To calculate the confusion 
# matrix a bunch of samples are run through the network with evaluate(), 
# which is the same as train() minus the backprop.
#
#    Reference(s):
#        https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Function to just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_training_sample()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure(figsize=(9.0,7.5))
ax = fig.add_subplot(111)
plt.subplots_adjust(top=0.802, bottom=0.060, left=0.165, right=0.983)
cax = ax.matshow(confusion)
cbar = fig.colorbar(cax)
cbar.set_label('Fraction of Correct Predictions', fontsize=ptextmedium, labelpad=22, rotation=270)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90, fontsize=ptextsmall)
ax.set_yticklabels([''] + all_categories, fontsize=ptextsmall)

ax.tick_params(axis='y', right=True)

ax.set_title("Confusion Matrix (N="+str(n_iters)+")", fontsize=ptextbig, pad=70)
ax.set_xlabel("Predicted", fontsize=ptextmedium, labelpad=15)
ax.set_ylabel("Actual", fontsize=ptextmedium)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


plt.show(block=False)


if VERBOSE == True:
    print("\n")

    insample = ["Albert", "Bergmann", "Tokyo"]
    
    for s in insample:
        input = line2tensor(s)
        output = evaluate(input)

        print('Sample input: ',s)
        print('Sample output:',output)
        print('Every item in the output tensor is the likelihood of that category.\n')

        print(category_from_output(output),'\n')


### SAVING (AND RESTORING) THE TRAINED RNN ###
# There are two main approaches for serializing and restoring a model.
#
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
# However in this case, the serialized data is bound to the specific 
# classes and the exact directory structure used, so it can break in 
# various ways when used in other projects, or after some serious refactors.
#    Reference(s):
#        https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch

#torch.save(rnn.state_dict(), "namelang_trained.pth")
torch.save(rnn, "namelang_rnn_trained.pth")

# This line is so Python doesn't bail out before giving us a chance to 
# examine the plots
plt.show()

