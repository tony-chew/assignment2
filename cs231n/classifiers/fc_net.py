from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # initialise weights (1st layer dimensions to 2nd layer dimensions)
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)

        # initialise biases to 0
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # extract initialised parameters
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        # 1st layer forward pass
        # NOTE: relu_cache contains fc_cache, relu_cache = (input, weights, bias, scores_preRelU)
        X2, relu_cache = affine_relu_forward(X, W1, b1)

        # 2nd layer forward pass
        scores, relu_cache_2 = affine_relu_forward(X2, W2, b2)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # compute loss
        # get softmax loss, apply L2 regularisation
        loss, dsoftmax = softmax_loss(scores, y)

        L2 = np.sum(W1 * W1) + np.sum(W2 * W2)
        loss += 0.5 * self.reg * L2

        # compute gradient
        # backward pass of last layer (2nd)
        dx2, dw2, db2 = affine_relu_backward(dsoftmax, relu_cache_2)

        # backward pass of first layer (1st)
        dx, dw, db = affine_relu_backward(dx2, relu_cache)

        # Put values into grads dictionary
        grads['W1'] = dw + self.reg * W1
        grads['W2'] = dw2 + self.reg * W2
        grads['b1'] = db
        grads['b2'] = db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # initialise dimensions through the layers (hidden_dims = list of layers)
        layers_dims = np.hstack([input_dim, hidden_dims, num_classes])

        # initialise weights and biases in a for loop according to number of layers
        for i in range(self.num_layers):
            self.params['W'+ str(i + 1)] = weight_scale * np.random.randn(layers_dims[i], layers_dims[i + 1])
            self.params['b'+ str(i + 1)] = np.zeros(layers_dims[i + 1])

        # initialise gamma and beta in a for loop when batch/layer normalisation occurs
        # up to the 2nd last layer as last layer is just affine forward pass
        if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
            for i in range(self.num_layers - 1):
                self.params['gamma'+ str(i + 1)] = np.ones(layers_dims[i + 1])
                self.params['beta'+ str(i + 1)] = np.zeros(layers_dims[i + 1])
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # initialise caches
        caches = []

        # to keep moving through the loop
        x = X

        # initialise gamma, beta, bn_params to None if batchnorm does not occur
        gamma, beta, bn_params = None, None, None

        # forward pass in a for loop according up to 2nd last layer
        for i in range(self.num_layers - 1):
            # for each layer up to the 2nd last layer, perform affine + ReLU

            # extract weights and biases for that layer
            W = self.params['W'+ str(i + 1)]
            b = self.params['b'+ str(i + 1)]

            if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
                # extract gamma, bias and bn_params for that layer if batch/layer norm
                gamma = self.params['gamma'+ str(i + 1)]
                beta = self.params['beta'+ str(i + 1)]
                bn_params = self.bn_params[i]

            # conduct affine --> batch --> relu forward pass
            x, cache = affine_batch_relu_forward(x, W, b, self.normalization, gamma, beta, bn_params,
                                                 self.use_dropout, self.dropout_param)

            # append cache to caches
            caches.append(cache)

        # forward pass for the last layer (just affine)
        # extract weights and biases for last layer
        W = self.params['W'+ str(self.num_layers)]
        b = self.params['b'+ str(self.num_layers)]

        # conduct affine forward pass and append final cache to caches list
        scores, cache = affine_forward(x, W, b)
        caches.append(cache)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ### compute loss
        # softmax loss function 
        loss, dsoftmax = softmax_loss(scores, y)

        # add L2 regularisation according to each layer
        L2 = 0.0
        for i in range(self.num_layers):
            w = self.params['W'+ str(i + 1)]
            loss += 0.5 * self.reg * np.sum(w * w)

        ### compute gradients
        # backward pass for the last layer (just affine) and using the last caches term
        dx, dw, db = affine_backward(dsoftmax, caches[self.num_layers - 1])

        # put values into grads dictionary
        grads['W'+ str(self.num_layers)] = dw + self.reg * self.params['W'+ str(self.num_layers)]
        grads['b'+ str(self.num_layers)] = db

        # backward pass for the other layers going back
        for i in range(self.num_layers - 2, -1, -1):
            # this counts backwards from the second last layer
            cache = caches[i]
            dx, dw, db, dgamma, dbeta =  affine_batch_relu_backward(dx, cache, self.normalization,
                                                                    self.use_dropout)

            # put values into grads dictionary
            grads['W'+ str(i + 1)] = dw + self.reg * self.params['W'+ str(i + 1)]
            grads['b'+ str(i + 1)] = db

            # put values of gamma and beta into grads if batch/layer norm occurs
            if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
                grads['gamma'+ str(i + 1)] = dgamma
                grads['beta'+ str(i + 1)] = dbeta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

# helper layers to use in fully connected net class

# forward pass
def affine_batch_relu_forward(x, w, b, normalization, gamma, beta, bn_params, dropout, dropout_param):

    #affine layer
    fc_out, fc_cache = affine_forward(x, w, b)

    # batch layer, if not activated, return None cache and pass input through
    if normalization == 'batchnorm':
        bn_out, bn_cache = batchnorm_forward(fc_out, gamma, beta, bn_params)

    elif normalization == 'layernorm':
        bn_out, bn_cache = layernorm_forward(fc_out, gamma, beta, bn_params)
        
    else:
        bn_out = fc_out
        bn_cache = None

    # relu layer
    re_out, re_cache = relu_forward(bn_out)

    # dropout layer, if not activated, return None cache and pass input through 
    if dropout:
        dropout_out, dropout_cache = dropout_forward(re_out, dropout_param)

    else:
        dropout_out = re_out
        dropout_cache = None

    # form cache of all the layers
    cache = fc_cache, bn_cache, re_cache, dropout_cache

    return dropout_out, cache

# backward pass
def affine_batch_relu_backward(dout, cache, normalization, dropout):

    # extract individual cache
    fc_cache, bn_cache, re_cache, dropout_cache = cache

    # dropout backward pass
    if dropout:
        d_dropout = dropout_backward(dout, dropout_cache)

    else:
        d_dropout = dout

    # relu backward pass
    d_re = relu_backward(d_dropout, re_cache)

    # batch backward pass
    # initialise dgamma, dbeta to None in the case that normalisation does not occur
    dgamma, dbeta = None, None
    
    if normalization == 'batchnorm':
        dx, dgamma, dbeta = batchnorm_backward_alt(d_re, bn_cache)

    elif normalization == 'layernorm':
        dx, dgamma, dbeta = layernorm_backward(d_re, bn_cache)
        
    else:
        dx = d_re

    # affine backward pass
    dx, dw, db = affine_backward(dx, fc_cache)

    return dx, dw, db, dgamma, dbeta
        
