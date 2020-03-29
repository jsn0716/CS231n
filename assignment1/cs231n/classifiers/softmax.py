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
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        scores = X[i].dot(W) # 1D vector of scores
        scores -= np.max(scores) # Normalize for numerical stability
        exp_scores = np.exp(scores)
        exp_scores_sum = np.sum(exp_scores)
        loss += -scores[y[i]] + np.log(exp_scores_sum)
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += X[i] * (exp_scores[j] / exp_scores_sum - 1)
            else:
                dW[:, j] += X[i] * (exp_scores[j] / exp_scores_sum)

    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += 2 * reg * W
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    scores = X.dot(W) # 2D matrix of scores with shape (num_train, num_classes)
    scores -= np.max(scores, axis=1)[:, None] # Normalize scores for each row
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores, axis=1)
    
    loss += np.sum(-scores[np.arange(num_train), y] + np.log(exp_scores_sum))
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    grad_coeffs = np.zeros(scores.shape)
    grad_coeffs += exp_scores / exp_scores_sum[:, None]
    grad_coeffs[np.arange(num_train), y] -= 1
    dW += np.dot(X.T, grad_coeffs)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
