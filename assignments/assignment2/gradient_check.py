import numpy as np

def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    #print('check_g, orig_x befor',orig_x)
    #print('check_g, x befor',x)
    
    fx, analytic_grad = f(x)
    
    #print('check_g, orig_x after',orig_x)
    #print('check_g, x.shape',x.shape)
    #print('func',f(x)[0])
    #print('fx=',fx,'analityc_grad=',analytic_grad)
    
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    #print('analitical grad.shape',analytic_grad.shape)
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #print('it.shape=',it.shape)
    while not it.finished:
        ix = it.multi_index
        #print('ix',ix)
        #print('x[ix]',x[ix])
        analytic_grad_at_ix = analytic_grad[ix]
        #print('analitical_grad-at_ix',analytic_grad_at_ix)
        orig_x = x.copy()
        #print('orig_x',orig_x)
        #print('x.shape befor delta',x.shape)
        orig_x[ix]+=delta
        #print('x.shape after delta',x.shape)
        #print('orig_x[ix] delta +',orig_x[ix])
        fx_plus=f(orig_x)[0]
        #fx_plus=fx_plus_full[ix[0]]
        #print('fx__plus',fx_plus)
        orig_x = x.copy()
        orig_x[ix]-=delta
        #print('orig_x[ix] delta -',orig_x[ix])
        fx_minus=f(orig_x)[0]
        #print('fx_minus',fx_minus)
        
        divider=2*delta
        #print('divider',divider)
        #numeric_grad_at_ix = np.divide((fx_plus-fx_minus),divider)
        numeric_grad_at_ix = (fx_plus-fx_minus)/divider
        #print('numeric_grad_at_ix',numeric_grad_at_ix)
        #print('fx(ix)', fx[ix])

        # TODO compute value of numeric gradient of f to idx
        
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

def check_gradient_single(f, x, delta=1e-5, tol=1e-4):
    """
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    """
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float

    fx, analytic_grad = f(x)
    analytic_grad = analytic_grad.copy()

    assert analytic_grad.shape == x.shape
    
    # the test below is from assesment 1

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #print('it.shape=',it.shape)
    while not it.finished:
        ix = it.multi_index
        #print('ix',ix, 'type(ix)',type(ix))
        #print('x[ix]',x[ix])
        analytic_grad_at_ix = analytic_grad[ix]
        #print('analitical_grad-at_ix',analytic_grad_at_ix)
        orig_x = x.copy()
        #print('orig_x',orig_x)
        #print('x.shape befor delta',x.shape)
        orig_x[ix]+=delta
        #print('x.shape after delta',x.shape)
        #print('orig_x[ix] delta +',orig_x[ix])
        fx_plus=f(orig_x)[0]
        #fx_plus=fx_plus_full[ix[0]]
        #print('fx__plus',fx_plus)
        orig_x = x.copy()
        orig_x[ix]-=delta
        #print('orig_x[ix] delta -',orig_x[ix])
        fx_minus=f(orig_x)[0]
        #print('fx_minus',fx_minus)
        
        divider=2*delta
        #print('divider',divider)
        #numeric_grad_at_ix = np.divide((fx_plus-fx_minus),divider)
        numeric_grad_at_ix = (fx_plus-fx_minus)/divider
        #print('numeric_grad_at_ix',numeric_grad_at_ix)
        #print('fx(ix)', fx[ix])

        # the end of the test from assesment 1

        #print('numeric_grad_at_is', numeric_grad_at_ix)
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (
                  ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True


def check_layer_gradient(layer, x, delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for the input and output of a layer

    Arguments:
      layer: neural network layer, with forward and backward functions
      x: starting point for layer input
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    #print('input',x)
    x_in = x.copy()
    output = layer.forward(x)
    #print('output',output)
    #print('input_x after func',x)
    np.random.seed(10)
    output_weight = np.random.randn(*output.shape)
    #output_weight = np.ones(*output.shape)
    

    def helper_func(x):
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        #print('np.ones_like',np.ones_like(output))
        d_out = np.ones_like(output) * output_weight
        grad = layer.backward(d_out)
        #print('grad',grad)
        return loss, grad

    #helper_func(x)
    #return True  #just to check
    return check_gradient(helper_func, x_in, delta, tol)


def check_layer_param_gradient(layer, x,
                               param_name,
                               delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for the parameter of the layer

    Arguments:
      layer: neural network layer, with forward and backward functions
      x: starting point for layer input
      param_name: name of the parameter
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    param = layer.params()[param_name]
    #print('parms.vname', param_name)
    #print('parms.value', param.value)
    initial_w = param.value.copy()
    #print('initial_w', initial_w)
    #print('initial_w.shape',initial_w.shape)

    
    output = layer.forward(x)
    np.random.seed(10)
    #output_weight = np.random.randn(*output.shape)
    output_weight = np.ones_like(output)

    def helper_func(w):
        param.value = w
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        #print('loss',loss,'d_out',d_out)
        ##print('layer grad befor backward',param.grad)
        layer.backward(d_out)
        grad = param.grad
        #print('layer grad after backward',grad)
        return loss, grad

    return check_gradient(helper_func, initial_w, delta, tol)


def check_model_gradient(model, X, y,
                         delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for all model parameters

    Arguments:
      model: neural network model with compute_loss_and_gradients
      X: batch of input data
      y: batch of labels
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    params = model.params()
        
    for param_key in params:
        print("Checking gradient for %s" % param_key)
        param = params[param_key]
        initial_w = param.value.copy()
        #print('initial_w',initial_w)
        #print('initial_w.shape',initial_w.shape)

        def helper_func(w):
            param.value = w
            loss = model.compute_loss_and_gradients(X, y)
            #d_out = np.ones_like(output)
            #print('check model grad loss',loss)
            grad = param.grad.copy()
            #print('grad.shape',grad.shape)
            return loss, grad

        if not check_gradient(helper_func, initial_w, delta, tol):
            return False

    return True
