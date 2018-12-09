#Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import find_objects
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import xml.etree.ElementTree as ET
import csv

DEPTH = str(3)

#Load image
def object_detection_save(image, folder_to_save):
    img = cv2.imread('datasets/kaggle/train/' + image + '.jpg')
    
    #Detect edges
    edges = cv2.Canny(img,150,200) #Canny Edge Detection #200 250
    edge = edges
    
    #Apply Gausian blur
    edges = gaussian_filter(edges, sigma=3)
    gauss = edges
    
    #Separate objects (label them differently)
    #https://stackoverflow.com/questions/16937158/extracting-connected-objects-from-an-image-in-python
    edges, nr_objects = ndimage.label(edges) 
    
    #Find objects (find x-y coordinates of rectangular)
    objects = find_objects(edges)
    
    objects_with_sizes = []
    #Calculate size of objects
    for obj in objects:
        ymin = obj[0].start
        ymax = obj[0].stop
        xmin = obj[1].start
        xmax = obj[1].stop
        area = (ymax - ymin) * (xmax - xmin)
        if(area < 115): #skip small objects
            continue
        objects_with_sizes.append([xmin, ymin, xmax, ymax, area, True])
        

    #Sort
    #https://wiki.python.org/moin/HowTo/Sorting
    objects = sorted(objects_with_sizes, key=lambda obj: obj[4])

    #Select 20 largest elements
    objects = objects[-30:] #8

    #Merge (overlapping and close areas)
    for obj in objects:
        if(obj[5]):
            for o in objects:
                if(o[5]):
                    if(((o[0] < obj[0] and o[2] + 20 > obj[0]) or (o[0] - 20 < obj[2] and o[2] > obj[2])) and
                      ((o[1] < obj[1] and o[3] + 20 > obj[1]) or (o[1] - 20 < obj[3] and o[3] > obj[3])) and
                      o[4] < 60000 and obj[4] < 60000): #Limit maximum size
                        obj[0] = min(o[0], obj[0])
                        obj[1] = min(o[1], obj[1])
                        obj[2] = max(o[2], obj[2])
                        obj[3] = max(o[3], obj[3])
                        obj[4] = (obj[3] - obj[1]) * (obj[2] - obj[0])
                        o[5] = False #Set flag to zero
    
    ships = []
    #Load truth from xml
    tree = ET.parse('crop/annotation/' + image + '.xml')
    root = tree.getroot()
    for ship in root.iter('bndbox'):
        xmin = int(ship.find('xmin').text)
        ymin = int(ship.find('ymin').text)
        xmax = int(ship.find('xmax').text)
        ymax = int(ship.find('ymax').text)
        ships.append([xmin, ymin, xmax, ymax])
    
    text_file = open(folder_to_save, "a")
    
    #iterate on objects
    for idx, obj in enumerate(objects): 
        if(obj[5]):
            #objects coordinates
            xmin = obj[0]
            ymin = obj[1]
            xmax = obj[2]
            ymax = obj[3]
            
            ships_on_crop = []
            #check ships coordinates
            for ship in ships:
                sh_xmin = max(xmin, ship[0]) - xmin
                sh_ymin = max(ymin, ship[1]) - ymin
                sh_xmax = min(xmax, ship[2]) - xmin
                sh_ymax = min(ymax, ship[3]) - ymin
                if(sh_xmin > sh_xmax or sh_ymin > sh_ymax):
                    continue
                ships_on_crop.append([sh_xmin, sh_ymin, sh_xmax, sh_ymax])
            
            #Crop and save image
            if(len(ships_on_crop)):
                filename = image + "_" + str(idx) + ".jpg"
                cv2.imwrite("crop/images/" + filename, img[ymin:ymax, xmin:xmax])
                annotation([xmax-xmin, ymax-ymin], ships_on_crop, "crop/images/", filename)
                text_file.write(filename[0:-4] + "\n")
    text_file.close()



def annotation(size_of_image, ships, folder, filename):  #Create the XML file with the annotation data
    #The XML file's format is identical with the ssd_keras sample annotation
    '''
    <annotation>
    <folder>train</folder>
    <filename>FILENAME.jpg</filename>
    <size>
        <width>768</width>
        <height>768</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>ship</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>XMIN</xmin>
            <ymin>YMIN</ymin>
            <xmax>XMAX</xmax>
            <ymax>YMAX</ymax>
        </bndbox>
    </object>
    </annotation>
    '''
    
    #create the XML file with the same structure
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(size_of_image[0])
    ET.SubElement(size, "height").text = str(size_of_image[1])
    ET.SubElement(size, "depth").text = DEPTH

    ET.SubElement(annotation, "segmented").text = str(0)

    for ship in ships:
        object_ = ET.SubElement(annotation, "object")
        ET.SubElement(object_, "name").text = "ship"
        ET.SubElement(object_, "pose").text = "Unspecified"
        ET.SubElement(object_, "truncated").text = str(0)
        ET.SubElement(object_, "difficult").text = str(0)
        bndbox = ET.SubElement(object_, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(ship[0])
        ET.SubElement(bndbox, "ymin").text = str(ship[1])
        ET.SubElement(bndbox, "xmax").text = str(ship[2])
        ET.SubElement(bndbox, "ymax").text = str(ship[3])

    tree = ET.ElementTree(annotation)
    tree.write("crop/annotation/" + filename[0:-4] + ".xml")

    
with open('crop/train_list.txt', 'r') as f: #Load data from CSV
    reader = csv.reader(f)
    samples = list(reader)
    samples = samples[1:]
    f.close()
    for image in samples:
        object_detection_save(image[0], 'crop/train_crop.txt')
        
with open('crop/val_list.txt', 'r') as f: #Load data from CSV
    reader = csv.reader(f)
    samples = list(reader)
    samples = samples[1:]
    f.close()
    for image in samples:
        object_detection_save(image[0], 'crop/val_crop.txt')
