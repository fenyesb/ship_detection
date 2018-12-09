import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import find_objects
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import xml.etree.ElementTree as ET
import csv
from resizeimage import resizeimage

DEPTH = str(3)

#Load image
def object_detection_save(image, set_):
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
    objects = objects[-30:]

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
                folder = "ship"
            else:
                folder = "noship"
            filename = folder + "/" + image + "_" + str(idx) + ".jpg"
            if (ymax-ymin < 150 and xmax-xmin < 150):
                cv2.imwrite("smallships/" + set_ + "/" + filename, img[ymin:ymax, xmin:xmax])
            
    
with open('train_list.txt', 'r') as f: #Load data from CSV
    reader = csv.reader(f)
    samples = list(reader)
    #samples = samples[1:]
    f.close()
    for image in samples:
        object_detection_save(image[0], "train")

with open('val_list.txt', 'r') as f: #Load data from CSV
    reader = csv.reader(f)
    samples = list(reader)
    #samples = samples[1:]
    f.close()
    for image in samples:
        object_detection_save(image[0], "val")
