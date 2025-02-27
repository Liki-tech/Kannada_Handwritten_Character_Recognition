import cv2
import numpy as np
import os, os.path
import tkinter as Tkinter
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import os, os.path
import tkinter as tk
from tkinter import filedialog
from sklearn.externals import joblib  # Assuming the model is saved using joblib


import time
import sys
set1=1
set2=0
cc=1
array=[]
i=0
# Function to upload an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png")])
    if file_path:
        img = cv2.imread(file_path)
        cv2.imshow("Original Image", img)
        return img
    return None

# Load the trained model
model = joblib.load('path_to_your_trained_model.pkl')  # Update with the actual model path

def extract_features(im_bw):
    # Example feature extraction logic
    # Flatten the image and return as features
    features = im_bw.flatten()  # This is a simple example; you may want to implement more complex feature extraction
    return features

# Function to recognize characters
def recognize_characters(im_bw):

    # Implement character recognition logic here using the model
    # Convert the binary image to the required format for the model
    # Example: Extract features and predict using the model
    features = extract_features(im_bw)  # You need to implement this function
    predictions = model.predict(features)
    return predictions

# Main function to run the application
def main():
    img = upload_image()
    if img is not None:
        # Existing preprocessing code...
        recognized_text = recognize_characters(im_bw)  # Call the recognition function
        print("Recognized Text:", recognized_text)  # Display recognized text


cv2.imshow("original image",img) 
kernel = np.ones((5,5),np.float32)/25
#dst = cv2.filter2D(img,-1,kernel)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
time.sleep(1)

gray_image = (255-gray_image)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_image = (255 - gray_image)
        cv2.imshow('Greyscale Image', gray_image)

thresh = 127
time.sleep(1)

        im_bw = cv2.threshold(gray_image, 75, 255, cv2.THRESH_BINARY)[1]

#im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
kernel = np.ones((1,1),np.uint8)
#im_bw = cv2.erode(im_bw,kernel,iterations = 1)
#opening = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
cv2.imshow('Binary image',im_bw)
#print(im_bw)
#f = open("a.txt", "w")
#f.writelines(array)
#print(type(im_bw))
#im_bw=np.array([[0,0,0,0,1,0],[0,1,1,0,1,0],[0,0,1,0,1,0],[0,0,1,0,1,0],[0,0,0,0,1,0]])
#opening = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
#cv2.imshow('ima',im_bw)
row,column=im_bw.shape
print(row)
print(column)
for y in range(0,column):
    #print(y)
    for x in range(0,row):
	    #print(x)
		if(set1==1):
			#print("set1")
			if(im_bw[x,y]!=0):
				array.append(y)
				#print(y)
				set1=0
				set2=1
				break
		elif(set2==1):
			#print("set2")
			if(im_bw[x,y]!=0):
			    break
			elif(x==row-1):
				array.append(y)
				#print(y)
				set1=1
				set2=0
				break
print(array)
length=len(array)
if(length%2!=0):
	length=length-1
for p in range(0,length,2):
	crop = im_bw[0:row,array[p]:array[p+1]]
	kernel = np.ones((3,3),np.uint8)
	#crop = cv2.erode(crop,kernel,iterations = 1)
	name = '/Users/sandeepa/Documents/shreya/8th sem shreya/Testing/a/crop%d.jpg' % (cc)
	cc=cc+1
	cv2.imwrite(name,crop)
	#cv2.imshow("image",crop)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
path="/Users/sandeepa/Documents/shreya/8th sem shreya/Testing/a"
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
			name = '/Users/sandeepa/Documents/shreya/8th sem shreya/Testing/b/crop%d.jpg' % (cc)
			cc=cc+1
			cv2.imwrite(name,crop)
			#cv2.imshow("image",crop)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
