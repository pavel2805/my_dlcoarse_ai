import numpy as np
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy

def calc_accuracy_multiclass(train_X,train_y, test_X, test_y,num_folds,K):
    
    knn_classifier = KNN(k=K)
    knn_classifier.fit(train_X, train_y)
    predict = knn_classifier.predict(test_X)
    #rint('predicted ',predict)
    #print('real value',test_y)
    accuracy = multiclass_accuracy(predict, test_y)
    print("Accuracy: %4.2f" % accuracy)
    return accuracy
    

def culc_f1_score(train_folds_X,train_folds_y, val_X, val_y,num_folds,K):
   

    binary_train_mask = (train_folds_y == 0) | (train_folds_y == 9)
    binary_train_X = train_folds_X[binary_train_mask]   #test
    #print('binary_train_X (new data set)=', binary_train_X.shape) #expect new_size (~121), 32,32,3

    binary_train_y_test = train_folds_y[binary_train_mask]  
    #print('binary_train_y_test shape =', binary_train_y_test.shape) 
#print('binary_train_y_test[0-10, new lavel set]', binary_train_y_test[0:10]) #expect 0s and 9s
#print('binary_train_y_test[0] type', type(binary_train_y_test[0])) #expect numpy.uint8

    binary_train_y = train_folds_y[binary_train_mask] == 0   
#print('binary_train_y shape =', binary_train_y.shape) #expect 121,
#print('binary_train_y[0:10', binary_train_y[:10]) #extect Folse, True



    binary_test_mask = (val_y == 0) | (val_y == 9)

    binary_test_X = val_X[binary_test_mask]
#print('binary_test_X.shape =', binary_test_X.shape) #expect !16
    binary_test_y = val_y[binary_test_mask] == 0

# Reshape to 1-dimensional array [num_samples, 32*32*3]
#print('binary_train_x shape befor =', binary_train_X.shape) #expect 161,32,32,3
    binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
#print('binary_train_X.shape[0]',binary_train_X.shape[0]) #expect 121
#print('binary_train_x shape after =', binary_train_X.shape) #expect 121,32*32*3 = 3072

    binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)
#print('binary_test_x shape after =', binary_test_X.shape) #expect 16,32*32*3 = 3072

#print('------------classify ')
# Create the classifier and call fit to train the model
# KNN just remembers all the data
    knn_classifier = KNN(k=K)
    knn_classifier.fit(binary_train_X, binary_train_y)


#print('----------------calculate the dists, no loops ')

    dists = knn_classifier.compute_distances_no_loops(binary_test_X)
    #print(dists)
#print(dists.shape)
    assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

#print('----------------calculate the time ')

# Lets look at the performance difference
#%timeit knn_classifier.compute_distances_two_loops(binary_test_X)
#%timeit knn_classifier.compute_distances_one_loop(binary_test_X)
#%timeit knn_classifier.compute_distances_no_loops(binary_test_X)

    prediction = knn_classifier.predict(binary_test_X)
    #print('real value=', binary_test_y)
#print('predicted ',prediction)

#print('----------------calculate metrics ')
    precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
#print("KNN with k = %s" % knn_classifier.k)
#print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1)) 
     
    return f1