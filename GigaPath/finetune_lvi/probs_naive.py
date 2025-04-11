
import numpy as np
import os
import h5py # type: ignore

"""
Save probabilities of the predicted LVI patches in a ASAP friendly format
"""

# split 
split= 'test'
model_arch= 'gigapath'

# optimal threshold
opt_thr= 0.95

# parent directory
if model_arch == 'swin':
    parent_dir = "/home/20215294/Data/LVI/swin_" + split + "/"
    probs_dir = "./eval_results/probs_" + split +"/all_probs_" + split + ".npy"
else:
    parent_dir = "/home/20215294/Data/LVI/gigapath_" + split + "/"
    probs_dir = "./eval_results/EVAL_not_freeze_aug_fixed/all_probs_" + split + ".npy"

d_0 = d_1 = 2048

os.makedirs(parent_dir, exist_ok = True)

# load the probabilities
data_prob = np.load(probs_dir, allow_pickle=True).item()

# labels directory
data_label = np.load('/home/20215294/Data/LVI/ground_truth_froc/all_labs.npy', allow_pickle=True).item()
data_coord = '/home/20215294/Data/LVI/patches_20x/patches'

"""
data has the following structure except for swin ones:
{slide_id:[coords, labels, probs]}
"""

# get the slide names
slide_ids = list(data_prob.keys())

recall_all =  []
fp_all = []

# iterate over the slides
for slide_id in slide_ids:

    # network predictions
    if 'probs' not in probs_dir.split('/')[2]:
        prob = np.array(data_prob[slide_id][2])
    else:
        prob = np.array(data_prob[slide_id][0])    
    
    # get the indices of the patches with high probabilities
    idx_ = np.where(prob > opt_thr)[0]

    # get the coordinates of the patches
    loc_all= []
    coords= np.array(h5py.File(data_coord + '/{0}.h5'.format(slide_id), 'r')['coords'][:])

    if len(idx_) > 0:
        for idx in idx_:
            r_0, r_1 = coords[idx]

            # save the coordinates 
            loc= [r_0, r_1, r_0 + d_0, r_1, r_0 + d_0, r_1 + d_1, r_0, r_1 + d_1]
            loc_all.append(loc)   

    # merge the overlapping patches
    for i in range(len(loc_all)):
        for j in range(i+1, len(loc_all)):
            # compute the intersection area
            xA = max(loc_all[i][0], loc_all[j][0])
            yA = max(loc_all[i][1], loc_all[j][1])
            xB = min(loc_all[i][4], loc_all[j][4])
            yB = min(loc_all[i][5], loc_all[j][5])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA) * max(0, yB - yA)

            if interArea > 0:
                # merge the patches
                xA = min(loc_all[i][0], loc_all[j][0])
                yA = min(loc_all[i][1], loc_all[j][1])
                xB = max(loc_all[i][4], loc_all[j][4])
                yB = max(loc_all[i][5], loc_all[j][5])

                loc_all[j]= [xA, yA, xB, yA, xB, yB, xA, yB]

                # remove the merged patch
                loc_all[i]= None

                break 
                
    # remove the None values
    loc_all= [loc for loc in loc_all if loc is not None]
    
    # second merge the overlapping patches
    for i in range(len(loc_all)):
        for j in range(i+1, len(loc_all)):
            # compute the intersection area
            xA = max(loc_all[i][0], loc_all[j][0])
            yA = max(loc_all[i][1], loc_all[j][1])
            xB = min(loc_all[i][4], loc_all[j][4])
            yB = min(loc_all[i][5], loc_all[j][5])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA) * max(0, yB - yA)

            if interArea > 0:
                # merge the patches
                xA = min(loc_all[i][0], loc_all[j][0])
                yA = min(loc_all[i][1], loc_all[j][1])
                xB = max(loc_all[i][4], loc_all[j][4])
                yB = max(loc_all[i][5], loc_all[j][5])

                loc_all[j]= [xA, yA, xB, yA, xB, yB, xA, yB]

                # remove the merged patch
                loc_all[i]= None

                break 

    # remove the None values
    loc_all= [loc for loc in loc_all if loc is not None]

    # save the coordinates of positively predicted patches
    np.save(parent_dir + '{0}.npy'.format(slide_id), loc_all)         

    # ground truth labels
    label = data_label[slide_id][0]

     # get the indices of the patches with 1 labels
    idx_ = np.where(label == 1)[0]

    # get the coordinates of the patches
    lab_all= []
    if len(idx_) > 0:

        for idx in idx_:
            r_0, r_1 = coords[idx]

            # save the coordinates 
            lab= [r_0, r_1, r_0 + d_0, r_1, r_0 + d_0, r_1 + d_1, r_0, r_1 + d_1]
            lab_all.append(lab)   

    # merge the overlapping patches
    for i in range(len(lab_all)):
        for j in range(i+1, len(lab_all)):
            # compute the intersection area
            xA = max(lab_all[i][0], lab_all[j][0])
            yA = max(lab_all[i][1], lab_all[j][1])
            xB = min(lab_all[i][4], lab_all[j][4])
            yB = min(lab_all[i][5], lab_all[j][5])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA) * max(0, yB - yA)

            if interArea > 0:
                # merge the patches
                xA = min(lab_all[i][0], lab_all[j][0])
                yA = min(lab_all[i][1], lab_all[j][1])
                xB = max(lab_all[i][4], lab_all[j][4])
                yB = max(lab_all[i][5], lab_all[j][5])

                lab_all[j]= [xA, yA, xB, yA, xB, yB, xA, yB]

                # remove the merged patch
                lab_all[i]= None

                break 
                
    # remove the None values
    lab_all= [lab for lab in lab_all if lab is not None]
    
    # compute recall
    tp= 0
    for lab in lab_all:
        for loc in loc_all:
            # compute the intersection area
            xA = max(lab[0], loc[0])
            yA = max(lab[1], loc[1])
            xB = min(lab[4], loc[4])
            yB = min(lab[5], loc[5])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA) * max(0, yB - yA)

            if interArea > 0:
                tp += 1
                break

    recall= tp/len(lab_all)
    recall_all.append(recall)

    # compute false positive numbers
    fp= 0
    for loc in loc_all:
        for lab in lab_all:
            # compute the intersection area
            xA = max(lab[0], loc[0])
            yA = max(lab[1], loc[1])
            xB = min(lab[4], loc[4])
            yB = min(lab[5], loc[5])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA) * max(0, yB - yA)

            if interArea > 0:
                break
        else:
            fp += 1        
    fp_all.append(fp)

# print the average recall and false positive numbers and standard deviation in one line with 2 decimal points
print('Recall: {0:.2f} ({1:.2f})'.format(round(np.mean(recall_all), 2), round(np.std(recall_all), 2)))
print('False Positive: {0:.2f} ({1:.2f})'.format(round(np.mean(fp_all), 2), round(np.std(fp_all), 2)))