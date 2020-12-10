from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compress each image information into row form
    dims = np.prod(x[0].shape)

    # reshape image into N x D form
    X = x.reshape(x.shape[0], dims)

    # compute forward pass
    out = X.dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compress each image information into row form
    dims = np.prod(x[0].shape)

    # reshape image input of layer into N x D form
    X = x.reshape(x.shape[0], dims)

    # compute dw
    dw = X.T.dot(dout)

    # compute dx, reshape into N x d_1, .. , d_k form
    dx_pre = dout.dot(w.T)
    dx = dx_pre.reshape(x.shape)

    # compute db
    db = np.sum(dout, axis = 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # NOTE: maximum compares element wise, max gets max of whole array
    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute dx
    # dout multiplied with (False, True) array of x condition 
    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    layernorm = bn_param.get('layer_norm', 0) # layernorm is 0 if otherwise not defined (for summation axis type)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # compute mean and variance
        mu = x.mean(axis = 0)
        var = x.var(axis = 0)

        # compute standard deviation
        std = np.sqrt(var + eps)

        # compute normalised input (out)
        xhat = (x - mu) / std
        out = gamma * xhat + beta

        # update running mean and variance (only if batchnorm exists)
        if layernorm == 0:    
            running_mean = momentum * running_mean + (1 - momentum) * mu
            running_var = momentum * running_var + (1 - momentum) * (std * std)

        # store values of importance into cache
        cache = [x, mu, std, gamma, xhat, layernorm]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        running_xhat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * running_xhat + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # extract variables from cache
    x, mu, std, gamma, xhat = cache[0], cache[1], cache[2], cache[3], cache[4]

    # extract dimensions of input/output
    N, D = dout.shape

    ### FROM OUTPUT
    # step 1. out --> intermediate addition --> beta
    dbeta = np.sum(dout, axis = 0)

    # step 2. out --> intermediate addition --> intermediate multiplication --> gamma
    dgamma = np.sum(xhat * dout, axis = 0)

    # step 3. out --> intermediate addition --> intermediate multiplication --> xhat
    dxhat = dout * gamma

    ### FROM xhat
    # step 4.1 xhat --> intermediate multiplication --> (x - mu)
    invstd = 1.0 / std
    dx_mu = dxhat * invstd

    # step 4.2 xhat --> intermediate multiplication --> inverse std
    dinvstd = np.sum(dxhat * (x - mu), axis = 0)

    # step 5. inverse std --> inverse --> std
    dstd = (-1. / (std**2)) * dinvstd

    # step 6. std --> sqrt --> variance
    dvar = 0.5 * (1. / std) * dstd

    # step 7. variance --> intermediate addition --> mean
    da = 1. / N * np.ones((N, D)) * dvar

    # step 8. mean --> intermediate square --> (x - mu) 2.
    dx_mu_2 = 2 * (x - mu) * da

    # step 9.1 (x - mu) --> intermediate subtraction --> x
    dx1 = dx_mu + dx_mu_2

    # step 9.2 (x - mu) --> intermediate subtraction --> mean
    dmu = -1. * np.sum(dx_mu + dx_mu_2, axis = 0)

    # step 10. mean --> x
    dx2 = 1. / N * np.ones((N, D)) * dmu
    dx = dx1 + dx2 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # extract variables from cache
    x, mu, std, gamma, xhat, axis = cache[0], cache[1], cache[2], cache[3], cache[4], cache[5]

    # extract dimensions of input/output
    N, D = dout.shape

    # compute db, dgamma
    dbeta = np.sum(dout, axis)
    dgamma = np.sum(xhat * dout, axis)

    # compute dx: refer to https://kevinzakka.github.io/2016/09/14/batch_normalization/
    # for finalised simplified derivation

    # firstly compute needed terms
    dxhat = dout * gamma

    # dx
    dx = ((1. / N) * (N * dxhat - np.sum(dxhat, axis = 0) - xhat * np.sum(dxhat * xhat, axis = 0))) / std
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # define values from ln_param, this initiates the train mode section of batchnorm_forward
    ln_param['mode'] = 'train'
    ln_param['layer_norm'] = 1 # set axis to 1 in bactchnorm computes

    # transpose the input values for batchnorm_forward function
    x = x.T
    gamma = gamma.reshape(-1, 1)
    beta = beta.reshape(-1, 1)

    # compute layernorm with transposed values, this normalises over the features (C) instead of images (N)
    out, cache = batchnorm_forward(x, gamma, beta, ln_param)

    # transpose output to get original dimensions back
    out = out.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # transpose the input values
    dout = dout.T

    # compute backward pass
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)

    # transpose output to get original dimensions back
    dx = dx.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # compute mask matrix (form True/False prob matrix based on p, then scaled)
        # NOTE: p is probability to keep a neuron active 
        mask = (np.random.rand(*x.shape) < p) / p

        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # initialise variable dimensions
    N, C, H, W = x.shape # N images, C channels (RGB), H x W image dims
    F, C, HH, WW = w.shape # F filters, C channels (RGB), HH x WW filter dims

    # extract padding and stride values from dictionary
    p = conv_param['pad']
    s = conv_param['stride']

    # padding of input data (pad index 2 and 3 (H and W))
    x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    # initialise output dimensions 
    H_out = int(((H - HH + 2 * p) / s) + 1)
    W_out = int(((W - WW + 2 * p) / s) + 1)
    index = []

    # forward pass
    for n in range(N):
        # for each image --> extract relevant input image index 
        x_pad_index = x_pad[n]

        for f in range(F):
            # for each filter number --> extract relevant weights and bias filter index
            w_index = w[f]
            b_index = b[f]

            for i in range(0, H_pad - HH + 1, s):
                # for each height range according to stride

                for j in range(0, W_pad - WW + 1, s):
                    # for each width range according to stride
                    # compute dot product, append to index 
                    test = np.sum((x_pad_index[:, i:i+HH, j:j+WW] * w_index)) + b_index
                    index.append(test)

    # reshape index to output dimensions 
    out = np.array(index).reshape(N, F, H_out, W_out) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # REFRENCES: https://www.linkedin.com/pulse/forward-back-propagation-over-cnn-code-from-scratch-coy-ulloa/
    # REFRENCES: https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c

    # initialise variable dimensions and cache
    x, w, b, conv_param = cache
    
    N, F, H_out, W_out = dout.shape
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    # extract padding and stride values from dictionary cache
    p = conv_param['pad']
    s = conv_param['stride']

    # initialise dx output dimensions
    dx = np.zeros(x.shape)  # N, C, H, W
    dw = np.zeros(w.shape)

    # padding of x and dx
    x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
    dx_pad = np.pad(dx, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    ### COMPUTE dw
    for f in range(F):
        # for each filter 

        # initalise height count
        H_index = 0
        
        for i in range(0, H_pad - HH + 1, s):
            # for each height range according to stride

            # initialise width count
            W_index = 0
            
            for j in range(0, W_pad - WW + 1, s):
                # for each width range according to stride
                # compute gradient
                dw[f] += np.sum(x_pad[:, :, i:i+HH, j:j+WW] * dout[:, f, H_index, W_index].reshape(N, 1, 1, 1), axis=0)

                # iterate over height and width indexes for next for loop compute
                W_index += 1

            H_index += 1

    ### COMPUTE dx
    
    for n in range(N):
        # for each image --> extract relevant image index
        dx_pad_index = dx_pad[n]
        
        for f in range(F):
            # for each filter and image --> extract relevant weight and dout index
            w_index = w[f]
            dout_index = dout[n, f]

            # initialise height count
            H_index = 0
        
            for i in range(0, H_pad - HH + 1, s):
                # for each height range according to stride

                # initialise width count
                W_index = 0
            
                for j in range(0, W_pad - WW + 1, s):
                    # for each width range according to stride
                    # compute gradient 
                    dx_pad_index[:, i:(i+HH), j:j+WW] += w_index * dout_index[H_index, W_index]
                
                    # iterate over height and width indexes for next for loop compute
                    W_index += 1
                    
                H_index += 1

    # get correct dimensions of dx
    dx = dx_pad[:, :, p:(H+p), p:(W+p)]

    ### COMPUTE db
    # summation of daxis to get into F filters
    db = np.sum(dout, axis = (0, 2, 3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # initialise variable dimensions
    N, C, H, W = x.shape

    # extract relevant variables from dictionary
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    s = pool_param['stride']

    # initialise out dimensions
    H_pool = int(((H - pool_height) / s) + 1)
    W_pool = int(((W - pool_width) / s) + 1)
    out = np.zeros((N, C, H_pool, W_pool))

    # forward pass
    # iterate through each height and width, summ through image wise and filter wise 

    # initialise height count
    H_index = 0
    
    for i in range(0, H - pool_height + 1, s):
        # for each height range according to stride

        # initialise width count
        W_index = 0
        
        for j in range(0, W - pool_width + 1, s):
            # for each width range according to stride
            # extract the max values of that particular pooling array 
            out[:, :, H_index, W_index] = np.max(x[:, :, i:i+pool_height, j:j+pool_width], axis = (2, 3))
            
            # iterate over height and width indexes for next for loop compute
            W_index += 1
            
        H_index += 1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # initialise variable dimensions and cache
    x, pool_param = cache
    N, C, H, W = x.shape
    _,_, H_out, W_out = dout.shape
    
    # extract relevant variables from dictionary
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    s = pool_param['stride']

    # initialise dx output dimensions
    dx = np.zeros(x.shape)  # N, C, H, W

    # backward pass

    for n in range(N):
        
        for c in range(C):
            # for each image and channel --> extract relevant image, dx and dout index
            x_index = x[n, c]
            dout_index = dout[n, c]
            dx_index = dx[n, c]

            # initialise height count
            H_index = 0
            
            for i in range(0, H - pool_height + 1, s):
                # for each height range according to stride

                # initialise width index
                W_index = 0
                
                for j in range(0, W - pool_width + 1, s):
                    # for each width range according to stride

                    # gather relevant x input sizing to pass back through
                    x_pass = x_index[i:(i+pool_height), j:(j+pool_width)]
                    
                    # use unravel_index with argmax to gather the indices of the max argument of that pooling size
                    max_ind = np.unravel_index(np.argmax(x_pass), x_pass.shape)

                    # put the relevant dout index value into dx according to max_ind
                    dx_index[i+max_ind[0], j+max_ind[1]] = dout_index[H_index, W_index]

                    # iterate over height and width indexes for next for loop compute
                    W_index += 1

                H_index += 1


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # initialise input dimension variables
    N, C, H, W = x.shape

    # IDEA: compute spatial batch statistics for each C feature over dimensions N, H and W
    # transpose x such that it becomes (N, H, W, C)
    x = np.transpose(x, (0, 2, 3, 1))

    # reshape such that it fits dimensions for batchnorm function input
    x = x.reshape(N * H * W, C)
    
    # conduct batch normalisation
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    # reshape out into (N, H, W, C)
    out = out.reshape(N, H, W, C)

    # transpose out such that (N, C, H, W)
    out = np.transpose(out, (0, 3, 1, 2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # initialise input dimension variables
    N, C, H, W = dout.shape

    # transpose dout such that it becomes (N, H, W, C)
    dout = np.transpose(dout, (0, 2, 3, 1))

    # reshape such that it fits dimensions for batchnorm_backward
    dout = dout.reshape(N * H * W, C)

    # conduct backward pass
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)

    # reshape dx into (N, H, W, C)
    dx = dx.reshape(N, H, W, C)

    # transpose dx such that (N, H, W, C)
    dx = np.transpose(dx, (0, 3, 1, 2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # initialise input dimension variables
    N, C, H, W = x.shape

    # initialise sizing due to parameter G
    size = (N * G, int(C / G * H * W))

    # resize input such that it becomes size.T shape
    x = np.transpose(x.reshape(size))

    # compute mean and variance
    mu = x.mean(axis = 0)
    var = x.var(axis = 0)

    # compute standard deviation
    std = np.sqrt(var + eps)

    # compute normalised input (out), then conduct transpose and reshaping back into N, C, H, W
    xhat = (x - mu) / std
    xhat = np.transpose(xhat).reshape(N, C, H, W)

    out = gamma * xhat + beta

    # store values of importance into cache
    cache = [gamma, std, mu, xhat, size]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
