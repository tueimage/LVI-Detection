import xml.etree.ElementTree as ET
import numpy as np
import glob
import os

# split
split = 'test'
model_arch= 'gigapath'

# Define directories for input and output data
if model_arch == 'swin':
    parent_dir_probs = "/home/20215294/Data/LVI/swin_" + split
    output_dir = "/home/20215294/Data/LVI/swin_" + split + "_anno"
else:
    parent_dir_probs = "/home/20215294/Data/LVI/gigapath_" + split
    output_dir = "/home/20215294/Data/LVI/gigapath_" + split + "_anno"

parent_dir_grounds = "/home/20215294/Data/LVI/ground_truth_asap"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of .npy files in the input directories
img_names_probs = glob.glob(parent_dir_probs + "/*.npy")

# Process each probability file
for img_name in img_names_probs:
    # Parse the template XML file
    tree = ET.parse('temp.xml')
    root = tree.getroot()
    
    # Load the probability points from the .npy file
    points_probs = np.load(img_name)
    
    # Add each point as an annotation in the XML
    for i, ps in enumerate(points_probs):
        processing = ET.Element("Annotation")
        processing.attrib['Name'] = str(i)
        processing.attrib['PartOfGroup'] = "predicted"
        processing.attrib['Color'] = "#ff0000"
        processing.attrib['Type'] = "Rectangle"
        
        # Append the annotation to the XML tree
        relevant_node = root.findall("Annotations")[0]
        relevant_node.append(processing)
        
        # Create a Coordinates sub-element
        processing_sub = ET.Element("Coordinates")
        relevant_node = root.findall("Annotations")[0].findall("Annotation")[-1]
        relevant_node.append(processing_sub)
        
        # Reshape the points and add each coordinate to the XML
        ps = np.array(ps).reshape(-1, 2)
        for j, xy in enumerate(ps):
            processing_sub_sub = ET.Element("Coordinate")
            processing_sub_sub.attrib['Order'] = str(j)
            processing_sub_sub.attrib['X'] = str(xy[0])
            processing_sub_sub.attrib['Y'] = str(xy[1])
            
            relevant_node = root.findall("Annotations")[0].findall("Annotation")[-1].findall("Coordinates")[0]
            relevant_node.append(processing_sub_sub)
    
    # Write the modified XML to the output directory
    tree.write(output_dir + '/' + '{0}.xml'.format(img_name.split('/')[-1][:-4]))

# Process each ground truth file
for img_name in img_names_probs:
    img_name = img_name.split('/')[-1]
    img_name = parent_dir_grounds + '/' + img_name

    # Load the ground truth points from the .npy file
    points_grounds = np.load(img_name)
    
    # Parse the corresponding XML file from the output directory
    tree = ET.parse(output_dir + '/' + '{0}.xml'.format(img_name.split('/')[-1][:-4]))
    root = tree.getroot()
    
    # Add each point as an annotation in the XML
    for i, ps in enumerate(points_grounds):
        processing = ET.Element("Annotation")
        processing.attrib['Name'] = str(i)
        processing.attrib['PartOfGroup'] = "ground truth"
        processing.attrib['Color'] = "#00aa00"
        processing.attrib['Type'] = "Rectangle"
        
        # Append the annotation to the XML tree
        relevant_node = root.findall("Annotations")[0]
        relevant_node.append(processing)
        
        # Create a Coordinates sub-element
        processing_sub = ET.Element("Coordinates")
        relevant_node = root.findall("Annotations")[0].findall("Annotation")[-1]
        relevant_node.append(processing_sub)
        
        # Reshape the points and add each coordinate to the XML
        ps = np.array(ps).reshape(-1, 2)
        for j, xy in enumerate(ps):
            processing_sub_sub = ET.Element("Coordinate")
            processing_sub_sub.attrib['Order'] = str(j)
            processing_sub_sub.attrib['X'] = str(xy[0])
            processing_sub_sub.attrib['Y'] = str(xy[1])
            
            relevant_node = root.findall("Annotations")[0].findall("Annotation")[-1].findall("Coordinates")[0]
            relevant_node.append(processing_sub_sub)
    
    # Write the modified XML to the output directory
    tree.write(output_dir + '/' + '{0}.xml'.format(img_name.split('/')[-1][:-4]))