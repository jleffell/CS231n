import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    margins = scores - correct_class_score + 1
    margins[y[i]]=0 # Only want to consider margin on wrong classes
    nmargins = (margins > 0).sum() # - 1
    for j in xrange(num_classes):
      if j == y[i]:
          dW[:,j] -= nmargins*X[i]
      else:
          margin = margins[j]
          if margin > 0:
              loss += margin
              dW[:,j] += X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss and gradient.
  loss += reg * np.sum(W * W)
  dW += reg * 2.0 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  delta = 1.0
  scores = np.dot(X,W)
    
  correct_class_score = scores[ range(num_train), y]
  margins = np.maximum(0, scores - correct_class_score.reshape(num_train,1) + delta)
  
  # Evaluate loss
  margins[range(num_train), y] = 0
  loss = np.sum(margins)

  # Evaluate gradient
  
  # Build matrix with ones on the j =/= y[i]'th column to add in single X[i] for each
  margins[margins>0]=1 # Set all positive margins for j =/= y[i] to 1
  # scale = np.count_nonzero(margins, axis=1) <- should use this if we had numpy > 10.12
  margins[range(num_train),y] = -(margins != 0).sum(1) # Set value for j == y[i] to number positive margins for each sample 
    
  # Assemble gradient
  dW = np.matmul(X.T,margins)    

  # Divide by number of samples
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss and gradient.
  loss += reg* np.sum(np.square(W))
  dW += reg * 2.0 * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
