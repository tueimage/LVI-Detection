
import numpy as np
from sklearn.metrics import auc

class MakeMetrics:
    '''
    A class to calculate metrics for multilabel classification tasks.
    
    Arguments:
    ----------
    metric (str): the metric to calculate. Default is 'froc'. Options are 'froc', 'auprc'.
    average (str): the averaging strategy. Default is 'micro'.
    '''
    def __init__(self, metric='auroc', average='micro'):
        self.metric = metric
        self.average = average

    def get_metric(self, labels: np.array, probs: np.array):
        '''Return the metric score based on the metric name.'''
        if self.metric == 'froc':
            sensitivity_list_treshold, FPavg_list_treshold = computeFROC(probs, labels)

            # prevent ValueError: At least 2 points are needed to compute area under curve
            if len(sensitivity_list_treshold) < 2:
                return 0
            else:
                area_froc = auc(np.array(FPavg_list_treshold), np.array(sensitivity_list_treshold))
                return area_froc
        
        elif self.metric == 'auprc':
            precision_list_treshold, sensitivity_list_treshold = computePRC(probs, labels)

            # prevent ValueError: At least 2 points are needed to compute area under curve
            if len(sensitivity_list_treshold) < 2:
                return 0
            else:
                area_prc = auc(np.array(sensitivity_list_treshold), np.array(precision_list_treshold))
                return area_prc
        else:
            raise ValueError('Invalid metric: {}'.format(self.metric))
    
    @property
    def get_metric_name(self):
        '''Return the metric name.'''
        if self.metric in ['froc', 'auprc']:
            if self.average is not None:
                return '{}_{}'.format(self.average, self.metric)
        else:
            return self.metric
        
    def __call__(self, labels: np.array, probs: np.array) -> dict:
        '''Calculate the metric based on the given labels and probabilities.
        Args:
            labels (np.array): the ground truth labels.
            probs (np.array): the predicted probabilities.'''
        # process the predictions
        if self.metric in ['froc', 'auprc']:
            if self.average is not None:
                return {self.get_metric_name: self.get_metric(labels, probs)}
        else:
            return {self.get_metric_name: self.get_metric(labels, probs)}


def calculate_multilabel_metrics(probs: np.array, labels: np.array, add_metrics: list=None) -> dict: 
    metrics = ['froc', 'auprc'] + (add_metrics if add_metrics is not None else [])
    results = {}
    for average in ['micro']: 
        for metric in metrics: 
            metric_func = MakeMetrics(metric=metric, average=average)
            results.update(metric_func(labels, probs))
    return results

def calculate_metrics_with_task_cfg(probs: np.array, labels: np.array, task_cfg: dict) -> dict:
    task_setting = task_cfg.get('setting', 'multi_class')
    add_metrics = task_cfg.get('add_metrics', None)

    if task_setting == 'multi_label':
        return calculate_multilabel_metrics(probs, labels, add_metrics)

def computeConfMatElements(thresholded_proba_map, ground_truth):
    P = np.count_nonzero(ground_truth)
    TP = np.count_nonzero(thresholded_proba_map*ground_truth)
    FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map*ground_truth)) 
    N = np.count_nonzero(1-ground_truth)                                
    return P,N,TP,FP


def computePRC(proba_map, ground_truth, nbr_of_thresholds= 40, range_threshold= [0, 1]):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}; or list of coordinates
    #nbr_of_thresholds: Interger. number of thresholds to compute to plot the FROC
    #range_threshold: list of 2 floats. Begining and end of the range of thresholds with which to plot the FROC  
    
    #OUTPUTS
    #sensitivity_list_treshold: list of average sensitivy over the set of images for increasing thresholds
    #precision_list_treshold: list of average FP over the set of images for increasing thresholds
            
    
    #define the thresholds
    threshold_list = (np.linspace(range_threshold[0],range_threshold[1],nbr_of_thresholds)).tolist()
    
    sensitivity_list_treshold = []
    precision_list_treshold = []
    FPavg_list_treshold = []
    #loop over thresholds
    for threshold in threshold_list:
        sensitivity_list_proba_map = []
        precision_list_proba_map = []
        FP_list_proba_map = []
        #loop over proba map
        for i in range(len(proba_map)):
            #probability of class_1
            prob= proba_map[i]  
            
            #threshold the proba map
            thresholded_proba_map = np.zeros(np.shape(prob))
            thresholded_proba_map[prob >= threshold] = 1                 
            ground_truth_i= ground_truth[i]
            
            #compute P, TP, FN for this threshold and this proba map
            P,_, TP,FP = computeConfMatElements(thresholded_proba_map, ground_truth_i)  
                 
            #check that ground truth contains at least one positive
            if TP+FP == 0:
                precision_list_proba_map.append(1)
            else:    
                precision_list_proba_map.append(TP/(TP+FP))

            sensitivity_list_proba_map.append(TP/P)
            FP_list_proba_map.append(FP)

        #average sensitivity and precision over the proba map, for a given threshold
        sensitivity_list_treshold.append(np.mean(sensitivity_list_proba_map))
        precision_list_treshold.append(np.mean(precision_list_proba_map))    
        FPavg_list_treshold.append(np.mean(FP_list_proba_map))

    #filter out Fpavg_list_treshold > 50
    sensitivity_list_treshold = [sensitivity_list_treshold[i] for i in range(len(FPavg_list_treshold)) if FPavg_list_treshold[i] <= 50]
    precision_list_treshold = [precision_list_treshold[i] for i in range(len(FPavg_list_treshold)) if FPavg_list_treshold[i] <= 50]
    return precision_list_treshold, sensitivity_list_treshold


def computeFROC(proba_map, ground_truth, nbr_of_thresholds= 40, range_threshold= [0, 1]):
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
            P,_,TP,FP = computeConfMatElements(thresholded_proba_map, ground_truth_i)       
            
            #append results to list
            FP_list_proba_map.append(FP)
            sensitivity_list_proba_map.append(TP*1./P)
            
        #print('\n\n') 
        #average sensitivity and FP over the proba map, for a given threshold
        sensitivity_list_treshold.append(np.mean(sensitivity_list_proba_map))
        FPavg_list_treshold.append(np.mean(FP_list_proba_map)) 


    # filter out Fpavg_list_treshold > 50
    sensitivity_list_treshold = [sensitivity_list_treshold[i] for i in range(len(FPavg_list_treshold)) if FPavg_list_treshold[i] <= 50]
    FPavg_list_treshold = [FPavg_list_treshold[i] for i in range(len(FPavg_list_treshold)) if FPavg_list_treshold[i] <= 50]
    return sensitivity_list_treshold, FPavg_list_treshold




if __name__ == '__main__':
    probs = [np.array([0.4, 0.3, 0.05, 0.04, 0.4, 0.8, 0.23])]

    # make labels into one-hot
    labels = [np.array([1, 1, 0, 0, 0, 1, 1])]
    print(calculate_multilabel_metrics(probs, labels))

