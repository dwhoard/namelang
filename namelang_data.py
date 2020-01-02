#!/usr/bin/env python3
# Reference(s): 
#     https://stackoverflow.com/questions/6908143/should-i-put-shebang-in-python-scripts-and-what-form-should-it-take


# namelang_data.py


# Defines functions for reading and cleaning the training data (and input) 
# to a Recurrent Neural Network (RNN) to predict the language origin 
# of a family name. 

# Requires:
#  namelang_model.py
#  namelang_train.py (to train the RNN)
#  namelang.py (to evaluate the trained RNN and perform a prediction)

# This Python script is run from within namelang_train and namelang - 
# it is not run independently.

# Adapted from the PyTorch tutorial "NLP FROM SCRATCH: CLASSIFYING NAMES 
# WITH A CHARACTER-LEVEL RNN" by Sean Robertson.
#     Reference(s):
#         https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#         https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

# The training data set is available at
#     https://download.pytorch.org/tutorial/data.zip

######################################################################

### READ THE TRAINING DATA ###

# Training data names are converted from Unicode to ASCII, to remove 
# any accent marks and provide pure ASCII transliterations of the names.

import glob
# glob = Unix style pathname pattern expansion
#    Reference(s): 
#        https://docs.python.org/3/library/glob.html

import os 
# os = Miscellaneous operating system interfaces
#    Reference(s): 
#        https://docs.python.org/3/library/os.html

import unicodedata
import string

import torch


### PRINT EXTRA DIAGNOSTIC INFO AS THE ROUTINE RUNS?
#VERBOSE = True
VERBOSE = False


# Define set of all allowed ascii characters
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Function to expand pathnames
def findfiles(path): return glob.glob(path)

# Example: find paths to training data files
if VERBOSE == True:
    print('\n*** Training data files:') 
    print(findfiles('data/names/*.txt'))


# Function to convert a Unicode string into plain ASCII
#    Reference(s): 
#        https://stackoverflow.com/a/518232/2809427
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Explanation: Unicode attempts to provide a unique code representing 
# each of all possible characters (glyphs) from all languages, whereas 
# ASCII encoding was originally based on the English alphabet plus some 
# control codes (like tab and newline). Accented characters are not 
# represented in ASCII, whereas in Unicode they can typically be 
# represented in two ways: as a single unique code (e.g., U+00f1 for 
# the lowercase letter ñ of the Spanish alphabet), or a combination 
# of a letter code followed by one or more "combining accent" codes 
# (e.g, U+006e U+0303 for the lowercase letter n from the English alphabet 
# followed by the combining tilde character). "Normalization" provides 
# a means of composing or decomposing a Unicode glyph code as a unique 
# code sequence in a specific order. This allows comparison of glyphs 
# (e.g., in text files) that might have multiple Unicode representations. 
# One of the normalization techniques is NFD ("Normalization Form Canonical 
# Decomposition").
#    Reference(s):
#        http://unicode.org/main.html

# Example: Unicode conversion
if VERBOSE == True:
    print('\n*** Example Text Conversion:')
    s = 'Ślusàrski'
    print('           Unicode=',s)
    c = unicodedata.normalize('NFD', s)
    print('Normalized Unicode=',c)
    print('             ASCII=', unicode2ascii('Ślusàrski'))


# Function to read a Unicode file and split it into lines converted to ascii
def readlines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode2ascii(line) for line in lines]


# Build the category_lines dictionary = a list of names per language
category_lines = {}
all_categories = []

# Loop over all training data files:
for filename in findfiles('data/names/*.txt'):
    # Extract individual filenames and truncate the filename suffix
    category = os.path.splitext(os.path.basename(filename))[0]
    # Add to list of categories (language names)
    all_categories.append(category)
    # Parse each file line-by-line, convert to ascii, and store in dictionary
    lines = readlines(filename)
    category_lines[category] = lines

# Define number of categories (languages)
n_categories = len(all_categories)


### CONVERT TRAINING DATA (NAMES) TO TENSORS ###
# To represent a single letter, we use a “one-hot vector” of size 
# (1 x n_letters). A one-hot vector is filled with 0s except for a 1 
# at the index of the current letter; e.g., "b" = <0 1 0 0 0 ...>.
#
# To make a word, we join a bunch of one-hot vectors into a 2D matrix, 
# (line_length x 1 x n_letters). The extra 1 dimension is needed because 
# PyTorch assumes everything is in batches - we’re just using a batch 
# size of 1 here.
#
#    Reference(s):
#        https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Function to find the letter index from all_letters; e.g., "a" = 0
def letter2index(letter):
    return all_letters.find(letter)

# Function to turn a line into a 2D matrix of one-hot vectors
def line2tensor(line):
    # Initialize the tensor iwth the right dimensions and all elements set to zero
    tensor = torch.zeros(len(line), 1, n_letters)
    # Set the value of the tensor element corresponding to each letter in line to one
    for li, letter in enumerate(line):
        tensor[li][0][letter2index(letter)] = 1
    return tensor

# Example: Converison to Tensor
if VERBOSE == True:
    print('\n*** Example Converison to Tensor')
    print('J =',letter2index('J'))
    print('Jones =',line2tensor('Jones').size())
    print('Jones =',line2tensor('Jones'),'\n')

