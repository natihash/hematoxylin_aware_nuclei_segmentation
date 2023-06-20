import cv2
import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
from skimage.draw import polygon
from skimage.draw import polygon_perimeter
from skimage.filters import threshold_otsu
from skimage.draw import disk

def xml_to_semantic(xml):
	tree = ET.parse(xml)
	root = tree.getroot()
	sema = np.zeros((1000, 1000))
	for i in range(1, len(root[0][1])):
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
		sema[rr, cc] = 255
	return sema

yy = xml_to_semantic("Monuseg_dataset/Annotations/TCGA-18-5592-01Z-00-DX1.xml")
cv2.imwrite("jhfj.png", 255*yy)
# print(yy)