from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    D, C = W.shape
    N = y.shape[0]
    loss = 0.0
    dW = np.zeros_like(W.T)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for n in range(N):
        loss -= np.log(np.exp(W.T[y[n]].dot(X[n]))/(np.exp(W.T.dot(X[n]))).sum()) - np.linalg.norm(W) * reg
    loss /= N
    # for c in range(C):
    #   for n in range(N):  
    #       dW[c] -= X[n].reshape((-1,1)).dot(((np.arange(C) == c) - np.exp(W.T[c].dot(X[n]))/(np.exp(W.T.dot(X[n]))).sum()).reshape((-1,1)).T)
    for c in range(C):
      for n in range(N):  
          dW[c] -= X[n]*((y[n] == c) - np.exp(W.T[c].dot(X[n]))/(np.exp(W.T.dot(X[n]))).sum()) - 2*W[:,c]*reg
    dW /= N
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW.T


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    D, C = W.shape
    N = y.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    loss -= (np.log(np.exp((X * W[:,y].T).sum(-1))/(np.exp((W.T.dot(X.T).T)).sum(-1)))).sum() - N * np.linalg.norm(W) * reg
    loss /= N

    dW = -X.T.dot(((np.logical_not(np.equal(np.full((N,C),np.arange(C)).T,np.full((C,N),y)))) - (np.exp((X * W[:,y].T).sum(-1))/(np.exp((W.T.dot(X.T).T)).sum(-1)))).T) - 2*W*reg*N
    dW /= N
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
