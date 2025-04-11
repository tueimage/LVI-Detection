
# This script plots the free-response curves for two different methods

import numpy as np  
from sklearn.metrics import auc # type: ignore
import matplotlib.pyplot as plt # type: ignore
font= {'size': 12, 'weight': 'bold', 'family': 'serif'}
plt.rc('font', **font)


#function to plot the FROC
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
    save_path_test = './eval_results/EVAL_not_freeze_aug_fixed/multiple_froc_val.pdf'


    # compute FROC for the GigaPath
    # FPavg_list_test_first = [0, 7.95, 11, 12.42, 13.74, 15.42, 17, 18.11, 19.53, 20.68, 21.42]
    # sensitivity_list_test_first = [0, 0.79, 0.79, 0.82, 0.84, 0.84, 0.84, 0.84, 0.85, 0.85, 0.85]

    FPavg_list_test_first = [0, 9.22, 12.67, 14.39, 16.11, 17.72, 18.72, 19.67, 20.56]
    sensitivity_list_test_first = [0, 0.72, 0.79, 0.81, 0.82, 0.88, 0.88, 0.91, 0.91]
    area_froc_first = auc(np.array(FPavg_list_test_first), np.array(sensitivity_list_test_first))

    #compute FROC for the Swin-Small
    # FPavg_list_test_second = [0, 5.63, 13.95, 21.79]
    # sensitivity_list_test_second = [0, 0.65, 0.75, 0.79]

    FPavg_list_test_second = [0, 4.72, 9.89, 15.44, 20.22]
    sensitivity_list_test_second = [0, 0.55, 0.62, 0.67, 0.74]
    area_froc_second = auc(np.array(FPavg_list_test_second), np.array(sensitivity_list_test_second))

    #plot FROC of the methods for the test set
    plotFROC(FPavg_list_test_first,sensitivity_list_test_first, area_froc_first, FPavg_list_test_second,sensitivity_list_test_second, area_froc_second, save_path_test)