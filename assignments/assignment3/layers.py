import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
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
    def __init__(self, n_input, n_output):
        np.random.seed(10)
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        np.random.seed(10)
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X_in = None

    def forward(self, X):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        self.X_in=X.copy()
        X_out = np.dot(X,self.W.value) + self.B.value
        return X_out

    def backward(self, d_out):
        # TODO copy from the previous assignment
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

        #raise Exception("Not implemented!")        
        #return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        np.random.seed(10)
        
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.W = Param(
            np.ones((filter_size, filter_size,in_channels, out_channels))
        )
        #self.W.value=np.ones_like((filter_size, filter_size,in_channels, out_channels)) #  just to ckeck
        self.B = Param(np.zeros((out_channels)))

        self.padding = padding
        self.fc_conv = FullyConnectedLayer(filter_size*filter_size*
                            in_channels,out_channels)
        self.fc_conv.W.value= np.reshape(self.W.value, (self.filter_size*self.filter_size*self.in_channels,self.out_channels))
        self.fc_conv.B.value= self.B.value
        self.X_in = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape
              
        if self.padding == 1:
            
            aa=X.copy()
            bb=np.insert(aa, aa.shape[1],0,axis=1)
            cc=np.insert(bb,0,0,axis=1)
            dd=np.insert(cc,cc.shape[2],0,axis=2)
            self.X_in=np.insert(dd,0,0,axis=2)  
        else:
            self.X_in=X.copy()
        #print('self.X_in,shape', self.X_in.shape)
                
        out_height = height-self.filter_size+1+2*self.padding
        out_width = width-self.filter_size+1+2*self.padding
        X_out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        #print('out_height',out_height)
        X_conv = np.zeros((batch_size, self.filter_size*self.filter_size*self.in_channels))
        W_conv = np.reshape(self.W.value, (self.filter_size*self.filter_size*self.in_channels,self.out_channels))
        #print('X_conv.shape', X_conv.shape)
        #print('W_conv.shape', W_conv.shape)
        #X_out = np.zeros((batch_size, self.filter_size, self.filter_size, self.out_channels))
        #print('X_out.shape', X_out.shape)
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        
       
        for oh in range(out_height):
            for ow in range(out_width):
                # TODO: Implement forward pass for specific location
                                           
                    #form X for this point inshape [batch_size, height*width*channels]
                for bs in range(batch_size):
                    #print('batch', bs,'point', oh,ow)      
                    ind=0
                    for ih in range(self.filter_size):
                        for iw in range(self.filter_size):
                            for ich in range(self.in_channels):
                                X_conv[bs,ind] = self.X_in[bs,oh+ih,ow+iw,ich]
                                #print('X_conv',X_conv[bs,ind])
                                ind+=1
                    #print('X_conv for current point',X_conv, oh,ow)
                    
                X_out_point = self.fc_conv.forward(X_conv)
                #make the output result with correct shape
                for i_bs in range(batch_size):
                    for i_och in range(self.out_channels):
                        X_out[i_bs,oh,ow,i_och]=X_out_point[i_bs,i_och]
        pass
        return X_out           
                
        
        #raise Exception("Not implemented!")


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X_in.shape
        #print('self.X_in.shape',self.X_in.shape)
        _, out_height, out_width, out_channels = d_out.shape
        #print('d_out.shape',d_out.shape)
        #print('d_out',d_out)
        #print('self.X_in.shape',self.X_in.shape)
        #print('self.X_in',self.X_in)

        dX_in = np.zeros_like(self.X_in)
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                dX_out_point=d_out[:,y,x,:]
                #print('d_out_point.shape',dX_out_point.shape)
                #print('d_out_point',dX_out_point)
                dX_in_point_flat = self.fc_conv.backward(dX_out_point)
                #print('d_in_point_flat.shape',dX_in_point_flat.shape)
                #print('dX_in_point_flat',dX_in_point_flat)
                #dX_in_point=np.reshape(dX_in_point_flat, (batch_size,height, width, channels))
                dX_in_point=np.reshape(dX_in_point_flat, (batch_size, self.filter_size,self.filter_size,self.in_channels))
                #print('dX_in_point.shape',dX_in_point.shape)
                #print('dX_in_point',dX_in_point)
                #print('self.filter_size',self.filter_size)
                #print('dX_in befor',dX_in)
                for fh in range(self.filter_size):
                    for fw in range(self.filter_size):                                           
                        dX_in[:,y+fh,x+fw,:]+=dX_in_point[:,fh,fw,:]
                #print('dX_in after',dX_in)
                pass
        
        if self.padding == 1:
            #print('dX_in.shape befor ',dX_in.shape)
            dX_in_no_padding =    dX_in[:,1:-1,1:-1,:]    
            #print('dX_in.shape after',dX_in.shape)
        else:
            dX_in_no_padding =    dX_in
            
        return dX_in_no_padding
        #raise Exception("Not implemented!")

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_in=X.copy()
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        #raise Exception("Not implemented!")
        out_height=int(height/self.stride)
        out_width=int(width/self.stride)
        #print('out_height', out_height)
        X_pool = np.zeros((batch_size,out_height,out_width,channels))
        for oh in range(out_height):
            for ow in range(out_width):
                #print('oh',oh,'ow',ow)
                X_pool_point=self.X_in[:,oh*self.stride:oh*self.stride+self.pool_size, ow*self.stride:ow*self.stride+self.pool_size,:]
                #print('X_pool_pint.shape',X_pool_point.shape)
                X_pool_max_h=np.max(X_pool_point,axis=1)
                X_pool_max_hw=np.max(X_pool_max_h,axis=1)
                X_pool[:,oh,ow,:]=X_pool_max_hw
                
        return X_pool                    

    def backward(self, d_out):       
        batch_size, height, width, channels = self.X_in.shape
        _, out_height, out_width, out_channels = d_out.shape
        #print('d_out.shape',d_out.shape)
        #print('d_out',d_out)
        #print('self.X_in.shape',self.X_in.shape)
        #print('self.X_in',self.X_in)

        dX_in = np.zeros_like(self.X_in)
        for oh in range(out_height):
            for ow in range(out_width):   
                X_pool_point=self.X_in[:,oh*self.stride:oh*self.stride+self.pool_size, ow*self.stride:ow*self.stride+self.pool_size,:]                      
                #print('out_point', oh, ow, 'X_pool_point.shape',X_pool_point.shape)
                #print('X_pool_point',X_pool_point)
                for bs in range(batch_size):
                    for ich in range(channels):
                        X_in_point=X_pool_point[bs,:,:,ich]
                        #print('X_in_pint.shape', X_in_point.shape)
                        ih_iw_maxind = np.unravel_index(np.argmax(X_in_point, axis=None), X_in_point.shape)
                        
                        dX_in[bs,oh*self.stride+ih_iw_maxind[0],ow*self.stride+ih_iw_maxind[1],ich]+=d_out[bs,oh,ow,ich]
                        #print('ih_iw_maxind[0]',ih_iw_maxind[0])
                        #print('ih_iw_maxind[1]',ih_iw_maxind[1])
                        #print('bs',bs,'ich',ich)
                        #print('dX_in[bs,ih_iw_maxind[0],ih_iw_maxind[1],ich]',aa)
                
        #print('dX_in', dX_in)   
        return dX_in
        
       
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_in=X.copy()

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        #raise Exception("Not implemented!")
        X_flat=np.reshape(self.X_in,(batch_size,height*width*channels))
        return X_flat

    def backward(self, d_out):
        # TODO: Implement backward pass
        #raise Exception("Not implemented!")
        dX_in=np.reshape(d_out,self.X_in.shape)
        return dX_in

    def params(self):
        # No params!
        return {}
