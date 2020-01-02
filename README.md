# namelang
An example Recurrent Neural Network to predict the language origin of a family name.

Adapted from the PyTorch tutorial "NLP FROM SCRATCH: CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN" by Sean Robertson.

Reference(s):
1. https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
1. https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

The training data set is available at https://download.pytorch.org/tutorial/data.zip

## Example:

% python namelang.py Schmidt
 
Language Origin (Probability):
  1. German (0.58)
  2. French (0.21)
  3. Scottish (0.10)
