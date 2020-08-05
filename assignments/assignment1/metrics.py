def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    arr_lenght=ground_truth.shape[0]
    
    tp=tn=fn=fp=0.0
    #print(arr_lenght, tp,tn,fn,fp)
    
    for i in range(arr_lenght):
        if (prediction[i] == True) and (ground_truth[i] == True):
            tp+=1
        if (prediction[i] == True) and (ground_truth[i] == False):
            fp+=1
        if (prediction[i] == False) and (ground_truth[i] == True):
            fn+=1
        if (prediction[i] == False) and (ground_truth[i] == False):
            tn+=1
    
    #print('tp',tp, 'fp',fp,'tn',tn,'fn',fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1 = 2*(precision*recall)/(precision+recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    arr_lenght=ground_truth.shape[0]
    
    tp=0.0
    #print(arr_lenght, tp,tn,fn,fp)
    
    for i in range(arr_lenght):
        if prediction[i] == ground_truth[i]:
            tp+=1
        
    
    #print('tp',tp, 'fp',fp,'tn',tn,'fn',fn)
    
    accuracy = (tp)/(arr_lenght)
    #f1 = 2*(precision*recall)/(precision+recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return accuracy

