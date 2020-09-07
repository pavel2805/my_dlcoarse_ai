def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
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

