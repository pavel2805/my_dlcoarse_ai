def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    arr_lenght=ground_truth.shape[0]
    
    tp=0.0
    #print(arr_lenght, tp,tn,fn,fp)
    
    for i in range(arr_lenght):
        if prediction[i] == ground_truth[i]:
            tp+=1
        
    
    #print('tp',tp, 'fp',fp,'tn',tn,'fn',fn)
    
    #print('TP',tp, 'arr_lengh',arr_lenght)
    accuracy = (tp)/(arr_lenght)
    # TODO: Implement computing accuracy
    #raise Exception("Not implemented!")

    return accuracy
