import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        #self.pred=[]
        self.n_output=n_output
        self.fc1 = FullyConnectedLayer(n_input,hidden_layer_size)
        self.relu1=ReLULayer()
        self.fc2=FullyConnectedLayer(hidden_layer_size,n_output)
        self.relu2=ReLULayer()
        # TODO Create necessary layers
        #raise Exception("Not implemented!")

    

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        
        #--------forward way ---------
        #print('self.fc1.W.value', self.fc1.W.value)
        #print('self.fc1.W.grad', self.fc1.W.grad)
        fc1_out=self.fc1.forward(X)
        #print('self.fc1.W.value after forward', self.fc1.W.value)
        
        #print('fc1_out',fc1_out)
        relu1_out=self.relu1.forward(fc1_out)
        #print('relu1_out',relu1_out)
        fc2_out=self.fc2.forward(relu1_out)
        #print('fc2_out',fc1_out)
        #relu2_out=self.relu2.forward(fc2_out)    
        loss,d_preds=softmax_with_cross_entropy(fc2_out, y)
        #self.pred=fc2_out
        #print('output.shape', output.shape)
        #print('loss',loss)
        loss_fc1_l2,grad_fc1_l2=l2_regularization(self.fc1.W.value,self.reg)
        #print('loss fc1 l2',loss_fc1_l2)
        loss_fc2_l2,grad_fc2_l2=l2_regularization(self.fc2.W.value,self.reg)
        #print('loss fc2 l2',loss_fc2_l2)
        loss_total=loss+loss_fc1_l2+loss_fc2_l2
         
        #loss_total=loss  
        
        #-------backward way -----------
        #d_out = np.ones_like(output)
        #print('d_preds.shape',d_preds.shape)
        d_out=d_preds
        #print('self.fc1.W.grad befor backward', self.fc1.W.grad)
        #print('d_out befor backward',d_out)
        #relu1_dX=self.relu1.backward(d_out)
        #relu2_dX=self.relu2.backward(d_out)
        fc2_dX=self.fc2.backward(d_out)
        #print('self.fc2.W.grad',self.fc2.W.grad)
        relu1_dX=self.relu1.backward(fc2_dX)
        fc1_dX=self.fc1.backward(relu1_dX)
        self.fc1.W.grad+=grad_fc1_l2
        self.fc2.W.grad+=grad_fc2_l2
        
        return loss_total

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        
        
        fc1_out=self.fc1.forward(X) 
        relu1_out=self.relu1.forward(fc1_out)        
        fc2_out=self.fc2.forward(relu1_out)        
        relu2_out=self.relu2.forward(fc2_out)
        output = fc2_out
        pred=np.argmax(output,axis=1)
        
        #pred= np.argmax(self.pred,axis=1)
        #print('pred_1', pred_1)
                
        #raise Exception("Not implemented!")
        return pred

    def params(self):
        #result{}
        result = {'W2':self.fc2.W, 'B2':self.fc2.B, 'B1':self.fc1.B, 'W1':self.fc1.W}
        
        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")

        return result
