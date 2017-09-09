import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax(z,j):
    return np.divide(np.exp(z[j]),np.sum(np.exp(z)))

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
      xi = X[i]
      f = xi.dot(W)
      f -= np.max(f)
      loss -= np.log(softmax(f,y[i])) 
      dW[:,y[i]] -= xi
      for j in xrange(num_classes):
          dW[:,j] += softmax(f,j)*xi
       
  loss /= num_train
  dW /= num_train
    
  loss += reg*np.sum(W*W)
  dW += reg*2.0*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  f = np.dot(X,W)
  f -= np.amax(f, axis=1).reshape(num_train,1)
    
  loss = -np.sum(np.log(np.divide(np.exp(f[range(num_train),y]),np.sum(np.exp(f),axis=1))))
  
  # Scale each sample by the softmax of f
  A = np.divide(np.exp(f),np.sum(np.exp(f),axis=1).reshape((num_train,1)))

  # Subtract off X[i] for correct class 
  A[range(num_train),y] -= 1
  
  dW = np.matmul(X.T,A)
    
  # Normalize
  loss /= num_train
  dW /= num_train
  
  # Regularization contribution
  loss += reg*np.sum(W*W)
  dW += reg*2.0*W
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

