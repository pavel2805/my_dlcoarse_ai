import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    #print('predictions',predictions)
    
    orig_predictions=predictions.copy()
    #print(orig_predictions.ndim)
    if np.ndim(orig_predictions)==1:
        orig_predictions=orig_predictions.reshape(1,-1)
        #print(orig_predictions.shape)
    probs=np.zeros(orig_predictions.shape)
    for i in range(orig_predictions.shape[0]):
        orig_predictions[i]-=np.max(orig_predictions[i])
        sum_exp=np.sum(np.exp(orig_predictions[i]))
        probs[i] = np.divide(np.exp(orig_predictions[i]),sum_exp)
    
    #orig_predictions-=np.max(orig_predictions)
    #print('orig_predictions', orig_predictions)
    
    #predictions-=np.max(predictions)
    #sum_exp=np.sum(np.exp(orig_predictions))
    #print('sum_exp',sum_exp)
    #probs = np.divide(np.exp(orig_predictions),sum_exp)
    #print('probs',probs)
    return probs
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    for i in range(probs.shape[0]):
        loss=-np.log(probs[i][target_index[i]])
    
    return loss
    raise Exception("Not implemented!")


def softmax_with_cross_entropy_batch(predictions, target_index):
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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
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
    #print('mean_loss',mean_loss)
    
    # calc gradient
    '''
    # ---- this code doesn't work' -----
    dloss=-1
    dp_log_probs = p_gold * dloss
    print('dp_log_probs',dp_log_probs)
    dlog_probs = (1/probs)* dp_log_probs
    print('dlog_probs',dlog_probs)
    
    i=0
    dprobs=np.zeros(probs.shape)
    while i < (orig_predictions.shape[0]):
        #print('index',i)
        dprobs[i] = probs[i]*np.sum(probs)
        i+=1
    print('dprobs',dprobs)
    dprediction = dprobs*dlog_probs
    print('dprediction',dprediction)
    '''
    #golden solution
    #print('probs',probs)
    #print('p_gold',p_gold)
    dprediction = (probs-p_gold)/batch_size
    
    
    #print('dprediction',dprediction)
    #dprediction[target_index]-=1
    
    #raise Exception("Not implemented!")

    return mean_loss, dprediction

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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    orig_predictions=predictions.copy()
    for i in orig_predictions:
        i-=np.max(i)
    #print('orig_prediction norm',orig_predictions)
    
    p_gold=np.zeros_like(orig_predictions)
    #print('p_gold',p_gold)
    #print('target_index',target_index)
    p_gold[target_index]= 1
    #print('p_gold ',p_gold)
    
    #softmax 
    pred_exp = np.exp(orig_predictions)
    #print('pred_exp',pred_exp)
    
    sum_exp=np.sum(pred_exp)
    #print('sum_exp',sum_exp)
    
    probs=np.zeros_like(orig_predictions)
    probs = pred_exp/sum_exp
    #print('probs',probs)
    
    log_probs = np.log(probs)
    #print('log_probs',log_probs)
    
    p_log_probs = p_gold * log_probs
    dp_log_probs = np.zeros(p_log_probs.shape)
    #print('p_log_probs',p_log_probs)
                       
    loss=-np.sum(p_log_probs)
    #print('loss',loss)
    
    # calc gradient
    '''
    # ---- this code doesn't work' -----
    dloss=-1
    dp_log_probs = p_gold * dloss
    print('dp_log_probs',dp_log_probs)
    dlog_probs = (1/probs)* dp_log_probs
    print('dlog_probs',dlog_probs)
    
    i=0
    dprobs=np.zeros(probs.shape)
    while i < (orig_predictions.shape[0]):
        #print('index',i)
        dprobs[i] = probs[i]*np.sum(probs)
        i+=1
    print('dprobs',dprobs)
    dprediction = dprobs*dlog_probs
    print('dprediction',dprediction)
    '''
    #golden solution
    dprediction = probs-p_gold
    #print('dprediction',dprediction)
    #dprediction[target_index]-=1
    
    #raise Exception("Not implemented!")

    return loss, dprediction

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
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")

    loss=reg_strength*np.sum(W**2)
    grad=2*reg_strength*W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dW_befor_dot = softmax_with_cross_entropy_batch(predictions, target_index)
    
    XT=np.transpose(X)
    dW = np.dot(XT,dW_befor_dot)
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None
        print('first W = none')

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            print('nake W')
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            #raise Exception("Not implemented!")
            for i in batches_indices:
                local_x=X[i]
                target_index=y[i]
                loss,dW=linear_softmax_l2(self.W, reg, local_x, target_index)
                self.W=self.W-learning_rate*dW
                #check_gradient(lambda w: linear_classifer.linear_softmax_l2(w, reg,X,y), self.W)

            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)
            #print('loss',loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        XW_sum=np.dot(X,self.W)
        #print('W_sum',XW_sum[:5])
        pred=np.argmax(XW_sum,axis=1)
        y_pred=pred

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
