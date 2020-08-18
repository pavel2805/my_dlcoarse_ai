import numpy as np

def linear_softmax_l2(W, reg_strength, X,  target_index):
    '''
    compuute total loss & dW
    is the sum of linear sofmax annd l2_reguulariisation
    '''
    loss_sm, dW_sm= linear_softmax(X, W, target_index)
    loss_l2,dW_l2=l2_regularization(W, reg_strength)
    loss=loss_sm+loss_l2
    dW=dW_sm+dW_l2
    return loss, dW

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
   
    
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")

    loss=reg_strength*np.sum(W**2)
    grad=2*reg_strength*W
    return loss, grad
    


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    
    batch_size=predictions.shape[0]
    orig_predictions=predictions.copy()
    for i in orig_predictions:
        i-=np.max(i)
    #print('orig_prediction norm',orig_predictions)
    
    p_gold=np.zeros_like(orig_predictions)
    #print('p_gold',p_gold)
    #print('target_index',target_index)
    for i in range(batch_size):
        p_gold[i][target_index[i]]= 1
    #print('p_gold ',p_gold)
    
    #softmax 
    pred_exp = np.exp(orig_predictions)
    #print('pred_exp',pred_exp)
    
    sum_exp=np.sum(pred_exp,axis=1)
    #print('sum_exp',sum_exp)
    
    probs=np.zeros_like(orig_predictions)
    for i in range(batch_size):
        probs[i] = pred_exp[i]/sum_exp[i]
    #print('probs',probs)
    
    log_probs = np.log(probs)
    #print('log_probs',log_probs)
    
    p_log_probs = p_gold * log_probs
    dp_log_probs = np.zeros(p_log_probs.shape)
    #print('p_log_probs',p_log_probs)
                       
    loss=-np.sum(p_log_probs,axis=1)
    #print('loss',loss)
    mean_loss=np.mean(loss)
    
    d_preds= (probs-p_gold)/batch_size
   
    return mean_loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        #raise Exception("Not implemented!")
        self.X_in=X.copy()  #to use it in backword
        X_out=X.copy()
        #X_out=X
        #print('self.X_temp in forward',self.X_temp)
        it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            #print(ix, X[ix])
            if X_out[ix] <0:
                X_out[ix]=0
            it.iternext()
        return X_out
            
        
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        X_in=self.X_in.copy()
        #print('X_in',X_in)
        dX=np.ones_like(X_in)
        it = np.nditer(X_in, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            #print(ix, X[ix])
            if X_in[ix] < 0:
                dX[ix]=0
            elif X_in[ix]==0:
                dX[ix] = 0.5
            else:
                dX[ix] = 1
                    
            it.iternext()
        #print('dX',dX)
        d_result=dX*d_out
        #print('d_out', d_out)
        #print('d_result',d_result)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output,seed=10):
        np.random.seed(seed)
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        np.random.seed(seed)
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        self.X_in=X.copy()
        X_out = np.dot(X,self.W.value) + self.B.value
        return X_out

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        #print('d_out;shape', d_out.shape)
        XT=np.transpose(self.X_in)
        dW = np.dot(XT,d_out)
        #print('dW', dW)
        self.W.grad=dW
        #print('dB.shape', self.B.value.shape)
        dB_t = np.sum(d_out, axis = 0)
        #print('dB_t',dB_t.shape)
        dB = np.array([dB_t])
        self.B.grad=dB
        #print('dB',dB.shape)
        W_T=np.transpose(self.W.value)
        dX_in = np.dot(d_out, W_T)
        #print('dX_in.shape',dX_in.shape)
        #raise Exception("Not implemented!")

        return dX_in

    def params(self):
        set={'W': self.W, 'B': self.B}
        return set
