import cv2
import numpy as np
import os, os.path


#Training dataset path
train_path=""


count_of_train_images=0
training_list=[]
labels=[]
window_row=5
window_col=5

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

for root, dirs, files in os.walk(train_path):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.splitext(fullpath)[1] == '.jpg':
			base=os.path.basename(f)
			os.path.splitext(base)
			FName=os.path.splitext(base)[0]
			FName=FName.split("_")[0]

			count_of_train_images=count_of_train_images+1

			image = cv2.imread(fullpath)
			cropped_img = image[5:94, 5:94]
			resized_train_image = cv2.resize(cropped_img,(int(60),int(60)))
			img_grayscale = cv2.cvtColor(resized_train_image, cv2.COLOR_BGR2GRAY)
			ret,img_binary = cv2.threshold(img_grayscale,127,255,cv2.THRESH_BINARY)
			
			image_binary_dilation = cv2.dilate(img_binary,kernel,iterations = 2) 
			img_binary_erosion = cv2.erode(image_binary_dilation,kernel,iterations = 1) 


			total_white_pixels=cv2.countNonZero(img_binary_erosion)
			#print("Total white pixels %d" %total_white_pixels)

			img_row=img_binary_erosion.shape[0]
			img_col=img_binary_erosion.shape[1]

			zones=0
			sum=0
			density_list=[]


			for r in range(0,img_row, window_row):
    				for c in range(0,img_col, window_col):
        				window = img_binary_erosion[r:r+window_row,c:c+window_col]

					window_white_pixels=cv2.countNonZero(window)
					#print("Window_white_pixels %d" %window_white_pixels)
					density=float(window_white_pixels)/(window_row*window_col)
					#print("Density white pixels %f" %density)
					density_list.append(density)
					#print(density_list)
					sum=sum+window_white_pixels
					zones=zones+1

			density_list=np.float32(density_list)
			density_array=np.asarray(density_list)

			training_list.append(density_array)
			labels.append(FName)



#print("Count of train images %d" %count_of_train_images)
training_matrix = np.asarray(training_list)
training_matrix=np.float32(training_matrix)


label_matrix=np.asarray(labels)
label_matrix=np.float32(label_matrix)


svm=cv2.SVM()
svm_params=dict(kernel_type=cv2.SVM_LINEAR,svm_type=cv2.SVM_C_SVC,C=2.67,gamma=5.383)

svm.train(training_matrix,label_matrix,params=svm_params)
svm.save('SVMZoningAccuracy.dat')


count_of_test_images=0
testing_list=[]
testing_labels=[]
k=0

#Testing dataset path
test_path=""

for root, dirs, files in os.walk(test_path):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.splitext(fullpath)[1] == '.jpg':
			base=os.path.basename(f)
			os.path.splitext(base)
			FileName=os.path.splitext(base)[0]
			FileName=FileName.split("_")[0]
			
			count_of_test_images=count_of_test_images+1

			test_data_image = cv2.imread(fullpath)
			cropped_test_image= test_data_image[5:94, 5:94]
			resized_image = cv2.resize(cropped_test_image,(int(60),int(60)))
			test_img_dimensions=resized_image.shape		
			test_img_grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)			
			test_ret,test_img_binary = cv2.threshold(test_img_grayscale,127,255,cv2.THRESH_BINARY)
			

			test_img_binary_dilation = cv2.dilate(test_img_binary,kernel,iterations = 2) 
			test_img_binary_erosion = cv2.erode(test_img_binary_dilation,kernel,iterations = 1) 

			cv2.imshow('test_image_binary',test_img_binary_erosion)

			test_density_list=[]
			test_sum=0
			test_zones=0
			for test_r in range(0,60, window_row):
    				for test_c in range(0,60, window_col):
        				window = test_img_binary_erosion[test_r:test_r+window_row,test_c:test_c+window_col]
					
					window_white_pixels=cv2.countNonZero(window)
					#print("Window_white_pixels of test image is %d" %window_white_pixels)
					density_test=float(window_white_pixels)/(window_row*window_col)
					#print("Density white pixels of test image is %f" %density_test)
					test_density_list.append(density_test)
					#print(density_list)
					test_sum=test_sum+window_white_pixels
					test_zones=test_zones+1

			test_density_list=np.float32(test_density_list)
			test_density_array=np.asarray(test_density_list)
			#print(test_density_array)

			testing_list.append(test_density_array)
			testing_labels.append([])
			testing_labels[k].append(FileName)
			k=k+1
			

print("Test images %d" %count_of_test_images)

testing_matrix = np.asarray(testing_list)
testing_matrix=np.float32(testing_matrix)

testing_label_matrix=np.asarray(testing_labels)
testing_label_matrix=np.float32(testing_label_matrix)
			
result=svm.predict_all(testing_matrix)

print(result)
#print(result.shape)

print(testing_label_matrix)
#print(testing_label_matrix.shape)

matches = (result==testing_label_matrix)
print(matches.shape)
print(matches)
correct = np.count_nonzero(matches)
accuracy = (correct*100.0)/result.size
print("Correct %d" %correct)
print("Accuracy %f" %accuracy)
print(accuracy)

cv2.waitKey(0)
cv2.destroyAllWindows()
 
			
