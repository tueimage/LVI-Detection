
# This script plots the free-response curves for two different methods

import numpy as np  
from sklearn.metrics import auc
import matplotlib.pyplot as plt
font= {'size': 12, 'weight': 'bold', 'family': 'serif'}
plt.rc('font', **font)

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
    FPavg_list_treshold_norm = []
    #loop over thresholds
    for threshold in threshold_list:
        sensitivity_list_proba_map = []
        FP_list_proba_map = []
        FP_list_proba_map_norm = []
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
            FP_list_proba_map_norm.append(FP*1./np.shape(ground_truth_i)[0])
            sensitivity_list_proba_map.append(TP*1./P)
            
        #print('\n\n') 
        #average sensitivity and FP over the proba map, for a given threshold
        sensitivity_list_treshold.append(np.mean(sensitivity_list_proba_map))
        FPavg_list_treshold.append(np.mean(FP_list_proba_map))    
        FPavg_list_treshold_norm.append(np.mean(FP_list_proba_map_norm))
    
    # filter out Fpavg_list_treshold > 50
    sensitivity_list_treshold = [sensitivity_list_treshold[i] for i in range(len(FPavg_list_treshold)) if FPavg_list_treshold[i] <= 50]
    FPavg_list_treshold = [FPavg_list_treshold[i] for i in range(len(FPavg_list_treshold)) if FPavg_list_treshold[i] <= 50]
    FPavg_list_treshold_norm = [FPavg_list_treshold_norm[i] for i in range(len(FPavg_list_treshold)) if FPavg_list_treshold[i] <= 50]  
    return sensitivity_list_treshold, FPavg_list_treshold, FPavg_list_treshold_norm

    
def plotFROC(x1,y1, auc1, x2, y2, auc2, save_path,threshold_list=None):
    plt.figure()
    # plot FROC for two methods together
    plt.plot(x1, y1, label='GigaPath, AUC= {0:.3f}'.format(auc1), color='blue', marker='o', linestyle='-')
    plt.plot(x2, y2, label='Swin_Small, AUC= {0:.3f}'.format(auc2), color='red', marker='o', linestyle='-')
    plt.legend(loc='lower right', shadow=True, fontsize='medium')
    plt.xlabel('FPavg', weight = 'bold')
    plt.ylabel('Sensitivity', weight = 'bold')
    plt.title('FROC for different methods', weight = 'bold')

    #annotate thresholds
    if threshold_list != None:
        #round thresholds
        threshold_list = [ '%.2f' % elem for elem in threshold_list ]            
        xy_buffer = None
        for i, xy in enumerate(zip(x1, y1)):
            if xy != xy_buffer:                                    
                plt.annotate(str(threshold_list[i]), xy=xy, textcoords='data')
                xy_buffer = xy
    
    plt.savefig(save_path, dpi=500, bbox_inches='tight')   
    

if __name__ == "__main__": 

    #parameters for the FROC
    save_path_test = './eval_results/EVAL_not_freeze_aug_fixed/multiple_froc_test.pdf'

    # get the labels and the probabilities for the first method
    data = np.load('./eval_results/EVAL_not_freeze_aug_fixed/all_probs_test.npy', allow_pickle=True).item()
    slide_ids = list(data.keys())
    labels_test = np.array([data[slide_id][1] for slide_id in slide_ids], dtype=object)
    probs_test = np.array([data[slide_id][2] for slide_id in slide_ids], dtype=object)

    #compute FROC for the first method
    sensitivity_list_test_first, FPavg_list_test_first, FPavg_list_test_first_norm = computeFROC(probs_test, labels_test)
    area_froc_first = auc(np.array(FPavg_list_test_first), np.array(sensitivity_list_test_first))

    # get the labels and the probabilities for the second method
    data = np.load('./eval_results/probs_test/all_probs_test.npy', allow_pickle=True).item()
    slide_ids = list(data.keys())
    probs_test = np.array([data[slide_id][0] for slide_id in slide_ids], dtype=object)
    # get the test labels (fixed)
    data = np.load('./eval_results/EVAL_not_freeze_aug_fixed/all_probs_test.npy', allow_pickle=True).item()
    labels_test = np.array([data[slide_id][1] for slide_id in slide_ids], dtype=object)

    #compute FROC for the second method
    sensitivity_list_test_second, FPavg_list_test_second, FPavg_list_test_second_norm = computeFROC(probs_test, labels_test)
    area_froc_second = auc(np.array(FPavg_list_test_second), np.array(sensitivity_list_test_second))

    #plot FROC of the methods for the test set
    plotFROC(FPavg_list_test_first,sensitivity_list_test_first, area_froc_first, FPavg_list_test_second,sensitivity_list_test_second, area_froc_second, save_path_test)