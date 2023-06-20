import cv2
import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
from skimage.draw import polygon
from skimage.draw import polygon_perimeter
from skimage.filters import threshold_otsu
from skimage.draw import disk
import ntpath

def val_ret(num):
    if num < 125:
        return 120
    else:
        return 240

def xml_to_weights(xml, hem_name):
	tree = ET.parse(xml)
	hema = cv2.imread(hem_name, 0)
	root = tree.getroot()
	weight = np.zeros((1000, 1000))
	for i in range(1, len(root[0][1])):
		temp = np.zeros((1000, 1000))
		x_pts = []
		y_pts = []
		pts = []
		for j in range(len(root[0][1][i][1])):
			x = int(float(root[0][1][i][1][j].attrib['Y']))
			if x >= 1000:
				x = 999
			y = int(float(root[0][1][i][1][j].attrib['X']))
			if y >= 1000:
				y = 999
			x_pts.append(x)
			y_pts.append(y)
			pts.append([y, x])
		x_pts = np.array(x_pts)
		y_pts = np.array(y_pts)
		pts = np.array(pts)
		rr, cc = polygon(x_pts, y_pts)
		temp[rr, cc] = 1
		temp2 = temp*hema
		weight[rr, cc] = val_ret(np.sum(temp2)/np.sum(temp))
	return weight

xmls = glob.glob("Monuseg_dataset/Annotations/*")
images = glob.glob("Monuseg_dataset/Tissue Images/*")
for i in range(len(xmls)):
	weight = xml_to_weights(xmls[i], images[i])
	cv2.imwrite("modified dataset/weights/"+ntpath.basename(name)[:-4]+".png", weight)
	print(i, end=" ")
	break