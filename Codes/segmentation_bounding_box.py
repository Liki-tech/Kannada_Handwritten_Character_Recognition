import cv2
import numpy as np
import os, os.path
import Tkinter
from tkFileDialog import askopenfilename
import tkFileDialog
import time
import sys
start_letter=1
end_letter=0
cc=1
array=[]
i=0

#image to be segmented
img = cv2.imread('9_2_26.jpg')
cv2.imshow("original image",img) 
time.sleep(1)

kernel = np.ones((5,5),np.float32)/25
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image = (255-gray_image)
cv2.imshow('Greyscale image',gray_image)
time.sleep(1)

thresh = 127
im_bw = cv2.threshold(gray_image, 75, 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((1,1),np.uint8)
cv2.imshow('Binary image',im_bw)
#print(im_bw)
#cv2.imshow('ima',im_bw)

row,column=im_bw.shape
print(row)
print(column)

for y in range(0,column):
    #print(y)
    for x in range(0,row):
	    #print(x)
		if(start_letter==1):
			if(im_bw[x,y]!=0):
				array.append(y)
				start_letter=0
				end_letter=1
				break
		elif(end_letter==1):
			if(im_bw[x,y]!=0):
			    break
			elif(x==row-1):
				array.append(y)
				start_letter=1
				end_letter=0
				break
print(array)
length=len(array)
if(length%2!=0):
	length=length-1
for p in range(0,length,2):
	crop = im_bw[0:row,array[p]:array[p+1]]
	kernel = np.ones((3,3),np.uint8)
	#crop = cv2.erode(crop,kernel,iterations = 1)

	#Path to folder where cropped images will be saved
	name = '/crop%d.jpg' % (cc)
	cc=cc+1
	cv2.imwrite(name,crop)
	#cv2.imshow("image",crop)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
path=""
set1=1
set2=0
cc=1
for root, dirs, files in os.walk(path):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.splitext(fullpath)[1] == '.jpg':
			array=[]
			base=os.path.basename(f)
			os.path.splitext(base)
			FName=os.path.splitext(base)[0]
			img = cv2.imread(fullpath)
			#cv2.imshow("image",img)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()
			kernel = np.ones((5,5),np.float32)/25
			dst = cv2.filter2D(img,-1,kernel)
			img2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
			#cv2.imshow('imag',img2)
			gray_image = (255-img2)
			#cv2.imshow('imae',gray_image)
			thresh = 127
			im_bw = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]
			#cv2.imshow('imake',im_bw)
			#print(im_bw[117])
			#im_bw=np.array([[0,0],[1,1],[0,1],[0,1],[0,0]])
			row,column=im_bw.shape
			#print(row)
			#print(column)
			for x in range(10,row):
				##print(x)
				for y in range(0,column):
					##print(y)
					if(set1==1):
						##print("set1")
						if(im_bw[x,y]==0):
							#print("in")
							#print(x)
							#print(y)
							array.append(x)
							##print(y)
							set1=0
							set2=1
							break
					elif(set2==1):
						##print("set2")
						if(im_bw[x,y]==0):
							break
						elif(y==column-1):
							#print("in")
							#print(x)
							#print(y)
							array.append(x)
							#print("hello")
							##print(y)
							set1=1
							set2=0
							break
			print(array)
			crop = im_bw[array[0]:array[1],0:column]
			kernel = np.ones((5,5),np.uint8)
			crop = cv2.erode(crop,kernel,iterations = 1)
			#resu = cv2.resize(crop,(int(89),int(89)))
			#print(resu.shape)
			cv2.imshow("Cropped image",crop)
			name = '/crop%d.jpg' % (cc)
			cc=cc+1
			cv2.imwrite(name,crop)
			#cv2.imshow("image",crop)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
						
				

					
