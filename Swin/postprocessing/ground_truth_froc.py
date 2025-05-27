
import json
import os
import glob
import numpy as np
from distance import get_distance # type: ignore
import h5py # type: ignore
import pandas as pd # type: ignore

"""
ground truth for FROC
"""

# Set the parent directory and other directories for coordinates and data
parent_dir = "/hpc/dla_patho/hassan_/LVI/ground_truth_froc/"
coord_dir = "/hpc/dla_patho/hassan_/patches_10x/patches/"
data_dir = "/hpc/dla_patho/skin/test/dataset/"

# Create a directory for results if it doesn't exist
os.makedirs(os.path.join(parent_dir), exist_ok=True)

# Set parameters
offset = 4
offset_neg = 1.5
size = np.array([2048, 2048])

# Get all image paths from the dataset directory
img_pathes = glob.glob('/hpc/dla_patho/skin/test/dataset/*')


# get the list of the annotations
anno = pd.read_csv("/home/dla_patho/hkeshvarikhojasteh/LVI/preprocessing/dataset/anno.txt", delimiter="\t")
map_ = pd.read_csv("/home/dla_patho/hkeshvarikhojasteh/LVI/preprocessing/dataset/map.txt")

# get the list of the images
img_names = map_['Img_Name']

# Iterate over each image path
for img_name in img_names:
    print(img_name)

    # get the all the LVI points for the image
    img_name_map = map_.loc[map_['Img_Name'] == img_name, "ID"].iloc[0]
    answer = anno.loc[anno['Image Name'] == img_name_map, "Answer"].iloc[0]

    # Convert JSON string to Python object
    annotations = json.loads(answer)

    # Extract centers of ellipse objects
    centers = [(obj["center"]["x"], obj["center"]["y"])
        for obj in annotations
        if obj["type"] == "ellipse"]

    # Read coordinates from the h5 file
    file = h5py.File(coord_dir + img_name + ".h5", "r")
    coords = np.array(file['coords'])
    labels = np.zeros(len(coords))

    # Calculate the center coordinates of the patches
    cc_x = coords[:, 0] + size[0] // 2
    cc_y = coords[:, 1] + size[1] // 2
    all_cc = [[cc_x[i], cc_y[i]] for i in range(len(cc_x))]

    # Iterate over each coordinate and extract close patches
    for ii in range(len(all_cc)):
        rr_x, rr_y = all_cc[ii]
        diss = get_distance([rr_x, rr_y], centers)
        rr = offset_neg * size[0] // offset
        if min(diss) < rr:
            labels[ii] = 1  

    # Save the labels to a file
    np.save(parent_dir + '{0}.npy'.format(img_name), labels)