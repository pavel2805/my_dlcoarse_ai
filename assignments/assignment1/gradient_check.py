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
    #print('check_g, x after',x)
    #print('func',f(x)[0])
    #print('fx=',fx,'analityc_grad=',analytic_grad)
    
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    #print('analytic grad.shape', analytic_grad.shape)
    #print('x.shape',x.shape)
    
    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    print('it.shape=',it.shape)
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

        # TODO compute value of numeric gradient of f to idx
        
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

def check_gradient_batch(f, x, delta=1e-5, tol = 1e-4):
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
    #print('check_g, x after',x)
    #print('func',f(x)[0])
    print('fx=',fx,'analityc_grad=',analytic_grad)
    
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    print('it.shape=',it.shape)
    while not it.finished:
        ix = it.multi_index
        print('ix',ix)
        #print('x[ix]',x[ix])
        analytic_grad_at_ix = analytic_grad[ix]
        print('analitical_grad-at_ix',analytic_grad_at_ix)
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
        numeric_grad_at_ix = (fx_plus[ix[0]]-fx_minus[ix[0]])/divider
        print('numeric_grad_at_ix',numeric_grad_at_ix)
        #print('fx(ix)', fx[ix])

        # TODO compute value of numeric gradient of f to idx
        
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True


        
