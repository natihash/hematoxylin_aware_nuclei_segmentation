import cv2
import glob
import xml.etree.ElementTree as ET
import numpy as np
from skimage.draw import polygon
import ntpath
import stainNorm_Vahadane

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

def xml_to_marker(xml):
	tree = ET.parse(xml)
	root = tree.getroot()
	mark = np.zeros((1000, 1000))
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
images = glob.glob("Monuseg_dataset/Tissue Images/*")

for i, name in enumerate(xmls):
	sema = xml_to_semantic(name)
	mark = xml_to_marker(name)
	cv2.imwrite("modified dataset/semas/"+ntpath.basename(name)[:-4]+".png", sema)
	cv2.imwrite("modified dataset/markers/"+ntpath.basename(name)[:-4]+".png", mark)
	print(i, end=" ")

for i, name in enumerate(images):
	image = cv2.imread(name)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	n = stainNorm_Vahadane.Normalizer()
	hem = n.hematoxylin(image)
	cv2.imwrite("modified dataset/hemas/"+ntpath.basename(name)[:-4]+".png", 255*hem)
	print(i, end=" ")