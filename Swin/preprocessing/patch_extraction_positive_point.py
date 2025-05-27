
import os
import glob
import numpy as np
from openslide import OpenSlide # type: ignore
import random
import pandas as pd # type: ignore
import json

"""
Extract positive LVI patches in a smarter way!!
"""

# define the directories
parent_dir = "/hpc/dla_patho/hassan_/LVI"
data_dir= "/hpc/dla_patho/skin/test/dataset/"

# create the directory for the positive patches if it does not exist
os.makedirs(os.path.join(parent_dir, "positive_images"), exist_ok = True)

# parameters
seed= 42
m= 2
number= 40
offset= 4
size= np.array([2048, 2048])

# set the seed
random.seed(seed)
count_all= 0

# get the list of the annotations
anno = pd.read_csv("dataset/anno.txt", delimiter="\t")
map_ = pd.read_csv("dataset/map.txt")

# get the list of the images
img_names = map_['Img_Name']

# loop over the images
for img_name in img_names:

    # create the directory for the image if it does not exist
    path = os.path.join(parent_dir, "positive_images" , img_name) 
    os.makedirs(path, exist_ok = True)
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
    
    l= len(centers)

    # loop over the annotations and extract the patches around the LVI point centers
    for ii in range(l):
            
            # load the image
            idx = 0
            im= OpenSlide(data_dir + img_name + '/' + img_name + '.ndpi')

            # get the center of the LVI point
            c_x = int(centers[ii][0])
            c_y = int(centers[ii][1])

            # check if enough patches are extracted
            while idx < number:

                # get the random coordinates
                r_x =  random.randrange(-size[0]//offset, size[0]//offset)
                r_y =  random.randrange(-size[1]//offset, size[1]//offset)

                # get the coordinates of the patch considering the center of the LVI point and random coordinates
                r_0 = c_x  - r_x - size[0]//2
                r_1 = c_y - r_y - size[1]//2

                # size of the patch at level m
                d_0 = size[0]//(2**m)
                d_1 = size[1]//(2**m)

                # read the patch
                im_patch= im.read_region((r_0, r_1), m, (d_0, d_1))
                im_patch= im_patch.convert('RGB')

                # save the patch
                im_patch.save(path+'/{0}.jpg'.format(count_all))
                count_all += 1
                idx += 1  
