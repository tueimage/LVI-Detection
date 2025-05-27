
import json
import os
import glob
import numpy as np
import pandas as pd # type: ignore


"""
ground truth for ASAP
"""
# Set the parent directory and other directories for coordinates and data
parent_dir = "/hpc/dla_patho/hassan_/LVI/ground_truth_asap/"
coord_dir = "/hpc/dla_patho/hassan_/patches_10x/patches/"
data_dir = "/hpc/dla_patho/skin/test/dataset/"


# Create a directory for results if it doesn't exist
os.makedirs(os.path.join(parent_dir), exist_ok=True)


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

    # Extract rect objects
    rects = [(obj["corner"]["x"], obj["corner"]["y"], obj["size"]["x"], obj["size"]["y"])
        for obj in annotations
        if obj["type"] == "rect"]

    # Iterate over each LVI center and save the coordinates in ASAP format
    loc_all = []
    for ii in range(len(rects)):
        rr_x, rr_y, d_0, d_1 = rects[ii]

        loc= [rr_x, rr_y, rr_x + d_0, rr_y, rr_x + d_0, rr_y + d_1, rr_x, rr_y + d_1]
        loc_all.append(loc)          

    # Save the coordinates to a file
    np.save(parent_dir + '{0}.npy'.format(img_name), loc_all)      
