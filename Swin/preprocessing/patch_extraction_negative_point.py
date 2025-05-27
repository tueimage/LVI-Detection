
import os
import glob
import numpy as np
from openslide import OpenSlide # type: ignore
import random
import h5py # type: ignore
import pandas as pd # type: ignore
import json
from distance import get_distance

"""
Extract negative LVI patches for each WSI in a smarter way!
"""

# define the directories
parent_dir = "/hpc/dla_patho/hassan_/LVI"
coord_dir= "/hpc/dla_patho/hassan_/patches_10x/patches/"
data_dir= "/hpc/dla_patho/skin/test/dataset/"

# create the directory if it does not exist
os.makedirs(os.path.join(parent_dir, "negative_images"), exist_ok = True)

# parameters
seed= 42
m= 2
number= 40
offset= 4
offset_neg= 1.5
number_neg= 4
size= np.array([2048, 2048])

# set the seed
random.seed(seed)
count_all= 0


# read the annotation file
anno = pd.read_csv("dataset/anno.txt", delimiter="\t")
map_ = pd.read_csv("dataset/map.txt")

# get the list of the images
img_names = map_['Img_Name']

# loop over the images
for img_name in img_names:

    # create the directory for the image if it does not exist
    path = os.path.join(parent_dir, "negative_images" , img_name)

    # ignore this image if the directory already exists
    if os.path.exists(path):
        continue
    else:
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
            
        # loop over the annotations and extract the negative patches around the LVI point centers
        for ii in range(len(centers)):

            # load the image
            idx = 0
            iteration = 0
            im= OpenSlide(data_dir+img_name+'/'+img_name+'.ndpi')

            # get the center of the LVI point
            c_x = int(centers[ii][0])
            c_y = int(centers[ii][1])

            # check if enough patches are extracted
            while iteration < 1000:                    
                # get the random coordinates
                r_x =  random.randrange(-2*size[0]//offset, 2*size[0]//offset)
                r_y =  random.randrange(-2*size[1]//offset, 2*size[1]//offset)

                # get the coordinates of the patch considering the center of the LVI point and random coordinates
                r_0 = c_x  - r_x
                r_1 = c_y - r_y

                # check if the patch is not close to the LVI point
                dis= get_distance([r_0, r_1], centers)
                rr= np.sqrt((offset_neg*size[0]//offset) ** 2.0 + (offset_neg*size[1]//offset) ** 2.0)

                if min(dis) > rr:

                    # size of the patch at level m
                    d_0 = size[0]//(2**m)
                    d_1 = size[1]//(2**m)

                    r_0 -= size[0]//2
                    r_1 -= size[1]//2

                    # read the patch
                    im_patch= im.read_region((r_0, r_1), m, (d_0, d_1))
                    im_patch= im_patch.convert('RGB')

                    # save the patch
                    im_patch.save(path+'/{0}.jpg'.format(count_all))
                    count_all += 1
                    idx += 1   

                if idx == number:
                    break     

                iteration += 1

        # extract the negative patches out of LVI points
        file= h5py.File(coord_dir + img_name + ".h5", "r")
        coords= np.array(file['coords'])            

        # get the coordinates of the patches after tessellating the image
        cc_x = coords[:, 0] + size[0]//2
        cc_y = coords[:, 1] + size[1]//2

        # number of out_patches to extract
        number_out = coords.shape[0]//number_neg

        # get the center of the patches
        all_cc = [[cc_x[i],cc_y[i]] for i in range(len(cc_x))]
        
        # check if enough patches are extracted
        idx = 0
        iteration = 0            
                    
        while iteration < 1000:                    
            # get the random coordinates
            rr_x =  random.randrange(0, im.dimensions[0])
            rr_y =  random.randrange(0, im.dimensions[1])

            # check if the patch is inside the image
            diss = get_distance([rr_x, rr_y], all_cc)    
            rrr = size[0]//2

            if min(diss) < rrr:
                
                # check if the patch is not close to the LVI point
                diss= get_distance([rr_x, rr_y], centers)
                rr= np.sqrt((offset_neg*size[0]//offset) ** 2.0 + (offset_neg*size[1]//offset) ** 2.0)

                if min(diss) > rr:
         
                    # size of the patch at level m
                    d_0 = size[0]//(2**m)
                    d_1 = size[1]//(2**m)

                    rr_x -= size[0]//2
                    rr_y -= size[1]//2

                    # read the patch
                    im_patch= im.read_region((rr_x, rr_y), m, (d_0, d_1))
                    im_patch= im_patch.convert('RGB')

                    # save the patch
                    im_patch.save(path+'/{0}_out.jpg'.format(count_all))
                    count_all += 1
                    idx += 1   

            if idx == number_out:
                break

            iteration += 1

