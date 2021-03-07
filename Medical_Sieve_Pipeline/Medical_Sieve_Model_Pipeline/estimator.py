from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np
import copy

def combinations(nums):
    ans = [[]]
    for row in nums:
        curr = []
        for combination in ans:
            for element in row:
                new_combination = copy.deepcopy(combination)
                new_combination.append(element)
                curr.append(new_combination)
        ans = curr
    return ans

def f1(matrix):
    precision = matrix[1][1]*1.0 / (matrix[0][1] + matrix[1][1])
    recall = matrix[1][1]*1.0 / (matrix[1][0] + matrix[1][1])
    return 2*((precision*recall)/(precision+recall))

def model_evaluation(val_preds, aspect_vectors, thresholds_set):
    mlb_aspect = MultiLabelBinarizer()
    mlb_aspect.fit([aspect_vectors.columns.values.tolist()]) 

    max_avg_f1 = 0
    max_hamming_score = 0
    max_exact_accuracy = 0
    max_fuzzy_accuracy = 0
    max_fuzzy_accuracy_pos = 0
    max_exact_accuracy_pos = 0
    max_avg_rocauc = 0
    max_confusion_matrix = None
    max_threshold_set = []

    for threshold_set in thresholds_set:
        predict_softmax = np.zeros(aspect_vectors.shape, dtype=int)
        for row_index, row in enumerate(val_preds):
            for index, each in enumerate(row):
                if each >= threshold_set[index]:
                    predict_softmax[row_index][index] = 1

        hamming_score = 1 - hamming_loss(predict_softmax, aspect_vectors) 
        num_fuzzy_match = 0
        num_fuzzy_match_pos = 0
        num_exact_match_pos = 0
        num_pos = 0
        for true, pre in zip(mlb_aspect.inverse_transform(aspect_vectors.values), mlb_aspect.inverse_transform(predict_softmax)):
            if len(true) != 0: 
                num_pos += 1
            intersect = set(pre).intersection(set(true))
            if (len(true)>0 and len(pre)>0 and len(intersect) > 0) or (len(true) == 0 and len(pre) == 0):
                num_fuzzy_match += 1
            if len(true)>0 and len(pre)>0 and len(intersect) > 0:
                num_fuzzy_match_pos += 1
            if len(true)>0 and len(pre)>0 and pre == true: 
                num_exact_match_pos += 1
        fuzzy_accuracy = num_fuzzy_match*1.0/len(predict_softmax)
        exact_accuracy = accuracy_score(predict_softmax, aspect_vectors)
        fuzzy_accuracy_pos =  num_fuzzy_match_pos*1.0/num_pos
        exact_accuracy_pos = num_exact_match_pos*1.0/num_pos

        class_f1 = []
        for aspect, confusion_matrix in zip(mlb_aspect.classes_, multilabel_confusion_matrix(aspect_vectors, predict_softmax)):
    #         print(aspect, ':',f1(confusion_matrix),'\n', confusion_matrix, '\n')
            class_f1.append(f1(confusion_matrix))
            
        rocauc_score = roc_auc_score(aspect_vectors, val_preds, 'weighted')
        if np.mean(class_f1) > max_avg_f1:
            max_threshold_set = threshold_set
            max_avg_f1 = max(max_avg_f1, np.mean(class_f1))
            max_hamming_score = hamming_score
            max_exact_accuracy = exact_accuracy
            max_fuzzy_accuracy = fuzzy_accuracy 
            max_exact_accuracy_pos = exact_accuracy_pos
            max_fuzzy_accuracy_pos = fuzzy_accuracy_pos
            max_avg_rocauc = rocauc_score
            max_confusion_matrix = multilabel_confusion_matrix(aspect_vectors, predict_softmax)
            
    print("threshold set:", max_threshold_set)
    print("Confusion Matrix for Each Aspect:\n" + "="*60)
    print(max_confusion_matrix)
    print("Result of Metrics for Evaluation:\n" + "="*60)
    print("Hamming score:", max_hamming_score)
    print("Exact accuracy:", max_exact_accuracy)
    print("Fuzzy accuracy:", max_fuzzy_accuracy)
    print("Exact accuracy (exclude negative):", max_exact_accuracy_pos )
    print("Fuzzy accuracy (exclude negative):", max_fuzzy_accuracy_pos)
    print("Average F1 Score: ", max_avg_f1)
    print("ROC AUC Score: ", max_avg_rocauc)
