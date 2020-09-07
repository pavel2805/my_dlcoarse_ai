import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
        self.conv1=ConvolutionalLayer(in_channels=input_shape[2], out_channels=conv1_channels, filter_size=3, padding=1)
        self.relu1=ReLULayer()
        self.maxpool1=MaxPoolingLayer(4, 4)
        self.conv2=ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=1)
        self.relu2=ReLULayer()
        self.maxpool2=MaxPoolingLayer(4, 4)
        self.flattener=Flattener()
        self.fc=FullyConnectedLayer(8, n_output_classes)
      


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        
        conv1_out=self.conv1.forward(X)
        relu1_out=self.relu1.forward(conv1_out)
        maxpool1_out=self.maxpool1.forward(relu1_out)
        conv2_out=self.conv2.forward(maxpool1_out)
        relu2_out=self.relu2.forward(conv2_out)
        maxpool2_out=self.maxpool2.forward(relu2_out)
        flattener_out=self.flattener.forward(maxpool2_out)
        fc_out=self.fc.forward(flattener_out)
        loss,d_preds=softmax_with_cross_entropy(fc_out, y)
        loss_total=loss
        
     
        #-------backward way -----------
        #d_out = np.ones_like(output)
        #print('d_preds.shape',d_preds.shape)
        d_out=d_preds
        fc_dX=self.fc.backward(d_out)
        flattener_dX=self.flattener.backward(fc_dX)
        maxpool2_dX=self.maxpool2.backward(flattener_dX)
        relu2_dX=self.relu2.backward(maxpool2_dX)
        conv2_dX=self.conv2.backward(relu2_dX)
        maxpool1_dX=self.maxpool1.backward(conv2_dX)
        relu1_dX=self.relu1.backward(maxpool1_dX)
        conv1_dX=self.conv1.backward(relu1_dX)
                     
        
        return loss_total


    def predict(self, X):
        # You can probably copy the code from previous assignment
        conv1_out=self.conv1.forward(X)
        relu1_out=self.relu1.forward(conv1_out)
        maxpool1_out=self.maxpool1.forward(relu1_out)
        conv2_out=self.conv2.forward(maxpool1_out)
        relu2_out=self.relu2.forward(conv2_out)
        maxpool2_out=self.maxpool2.forward(relu2_out)
        flattener_out=self.flattener.forward(maxpool2_out)
        fc_out=self.fc.forward(flattener_out)
                
        output = fc_out
        pred=np.argmax(output,axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result = {'WFC':self.fc.W, 'BFC':self.fc.B, 'BCV1':self.conv1.B, 'WCV1':self.conv1.W,'BCV2':self.conv2.B, 'WCV2':self.conv2.W} 

        return result
