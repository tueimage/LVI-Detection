
# This script computes the optimal threshold 

import numpy as np  
from sklearn.metrics import f1_score # type: ignore
from sklearn.metrics import roc_auc_score # type: ignore

def computeConfMatElements(thresholded_proba_map, ground_truth):
    P = np.count_nonzero(ground_truth)
    TP = np.count_nonzero(thresholded_proba_map*ground_truth)
    FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map*ground_truth))                                 
    return P,TP,FP


def computeFROC(proba_map, ground_truth, nbr_of_thresholds=40, range_threshold=[0, 1]):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}; or list of coordinates
    #nbr_of_thresholds: Interger. number of thresholds to compute to plot the FROC
    #range_threshold: list of 2 floats. Begining and end of the range of thresholds with which to plot the FROC  
    
    #OUTPUTS
    #sensitivity_list_treshold: list of average sensitivy over the set of images for increasing thresholds
    #FPavg_list_treshold: list of average FP over the set of images for increasing thresholds
    #threshold_list: list of thresholds
            
    
    #define the thresholds
    threshold_list = (np.linspace(range_threshold[0],range_threshold[1],nbr_of_thresholds)).tolist()
    
    sensitivity_list_treshold = []
    FPavg_list_treshold = []

    #loop over thresholds
    for threshold in threshold_list:
        sensitivity_list_proba_map = []
        FP_list_proba_map = []
        #loop over proba map
        for i in range(len(proba_map)):
            #probability of class_1
            #print('Threshold: {0:.2f}, Evaluating Performance on image: {1}'.format(threshold, names[i]))  
            prob= proba_map[i]  
            
            #threshold the proba map
            thresholded_proba_map = np.zeros(np.shape(prob))
            thresholded_proba_map[prob >= threshold] = 1                 
            ground_truth_i= ground_truth[i]
            
            #compute P, TP, and FP for this threshold and this proba map
            P,TP,FP = computeConfMatElements(thresholded_proba_map, ground_truth_i)       
            
            #append results to list
            FP_list_proba_map.append(FP)
            sensitivity_list_proba_map.append(TP*1./P)
            
        #print('\n\n') 
        #average sensitivity and FP over the proba map, for a given threshold
        sensitivity_list_treshold.append(np.mean(sensitivity_list_proba_map))
        FPavg_list_treshold.append(np.mean(FP_list_proba_map))    
      
    return sensitivity_list_treshold, FPavg_list_treshold, threshold_list

if __name__ == "__main__": 

    #parameters
    parent_dir_val = 'EVAL_not_freeze_aug_fixed'
    parent_dir_test = 'EVAL_not_freeze_aug_fixed'
    nbr_of_thresholds = 40
    range_threshold = [0, 1]

    # labels directory
    data_label = np.load('/home/20215294/Data/LVI/ground_truth_froc/all_labs.npy', allow_pickle=True).item()

    # load the probabilities
    data_prob = np.load('./eval_results/' + parent_dir_val + '/all_probs_val.npy', allow_pickle=True).item()


    # get the labels and the probabilities for each slide
    slide_ids = list(data_prob.keys())
    labels_val = np.array([data_label[slide_id][0] for slide_id in slide_ids], dtype=object)

    if 'probs' not in parent_dir_val:
        probs_val = np.array([data_prob[slide_id][2] for slide_id in slide_ids], dtype=object)
    else:    
        probs_val = np.array([data_prob[slide_id][0] for slide_id in slide_ids], dtype=object)

    # compute the maximum f1 score
    f1_all = []
    threshold_list = (np.linspace(range_threshold[0],range_threshold[1],nbr_of_thresholds)).tolist()
    for threshold in threshold_list:
        f1_scores = []
        for i in range(len(probs_val)):
            prob = probs_val[i]
            label = labels_val[i]
            f1_scores.append(f1_score(label, np.where(prob > threshold, 1, 0)))
        f1_all.append(np.mean(f1_scores))    

    # print the results   
    print(f'\n\nmax val f1: {max(f1_all):.2f}, opt threshold: {threshold_list[np.argmax(f1_all)]:.2f}')

    # test f1 score for the opt threshold
    data_prob = np.load('./eval_results/' + parent_dir_test + '/all_probs_test.npy', allow_pickle=True).item() 

    # get the labels and the probabilities for each slide     
    slide_ids = list(data_prob.keys())
    labels_test = np.array([data_label[slide_id][0] for slide_id in slide_ids], dtype=object)

    if 'probs' not in parent_dir_test:
        probs_test = np.array([data_prob[slide_id][2] for slide_id in slide_ids], dtype=object)
    else:
        probs_test = np.array([data_prob[slide_id][0] for slide_id in slide_ids], dtype=object)

    # compute the test f1 score
    opt_threshold = threshold_list[np.argmax(f1_all)]
    f1_scores = []
    for i in range(len(probs_test)):
        prob = probs_test[i]
        label = labels_test[i]
        f1_scores.append(f1_score(label, np.where(prob > opt_threshold, 1, 0)))

    # print the results
    print(f'test f1: {np.mean(f1_scores):.2f}\n\n')

    #compute FROC  
    sensitivity_list_val, FPavg_list_val, threshold_list = computeFROC(probs_val, labels_val)
    sensitivity_list_test, FPavg_list_test, threshold_list = computeFROC(probs_test, labels_test)

    # Print thresholds, sensitivities, and FPavg for validation
    for i in range(len(threshold_list)):
        print(f'Threshold_val: {threshold_list[i]:.2f}, Sensitivity_val: {sensitivity_list_val[i]:.2f}, FPavg_val: {FPavg_list_val[i]:.2f}')

    print("\n\n")

    # Print thresholds, sensitivities, and FPavg for testing
    for i in range(len(threshold_list)):
        print(f'Threshold_test: {threshold_list[i]:.2f}, Sensitivity_test: {sensitivity_list_test[i]:.2f}, FPavg_test: {FPavg_list_test[i]:.2f}')

    # compute & print the auc score for the test set
    auc_scores = []
    for i in range(len(probs_test)):
        prob = probs_test[i]
        label = labels_test[i]
        auc_scores.append(roc_auc_score(label, prob))

    print(f'test auc: {np.mean(auc_scores)}')
