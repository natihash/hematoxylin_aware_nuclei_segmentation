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

def xml_to_weights(xml):
	tree = ET.parse(xml)
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
		temp[rr, cc] = 255
		temp = temp.astype('uint8')
		temp = cv2.distanceTransform(temp, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
		temp = 1.0*(temp > np.amax(temp)/2)
		mark[temp>0] = 255
	return mark

xmls = glob.glob("Monuseg_dataset/Annotations/*")
for i, name in enumerate(xmls):
	weight = xml_to_weights(name)
	np.save("modified dataset/weights/"+ntpath.basename(name)[:-4]+".npy", weight)
	print(i, end=" ")