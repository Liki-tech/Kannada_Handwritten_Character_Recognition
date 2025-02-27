import cv2
import numpy as np
import os, os.path
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
unicode_File_Name= open("Unicode_machine_editable.txt","w+")

#training datset path
train_path=""


count_of_train_images=0
training_list=[]
training_labels=[]
window_row=5
window_col=5

for root, dirs, files in os.walk(train_path):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.splitext(fullpath)[1] == '.jpg':
			base=os.path.basename(f)
			os.path.splitext(base)
			train_file_name=os.path.splitext(base)[0]
			#print(FName)
			train_file_name=train_file_name.split("_")[0]
			
			count_of_train_images=count_of_train_images+1

			image = cv2.imread(fullpath)
			cropped_image = image[5:94, 5:94]
			resized_train_image = cv2.resize(cropped_image,(int(60),int(60)))
			
			image_grayscale=cv2.cvtColor(resized_train_image, cv2.COLOR_BGR2GRAY)
			ret,image_binary = cv2.threshold(image_grayscale,127,255,cv2.THRESH_BINARY) 
			

			image_row=image_binary.shape[0]
			image_col=image_binary.shape[1]
			total_white_pixels=cv2.countNonZero(image_binary)
			
			train_zones=0
			sum=0
			train_density_list=[]

			for train_row in range(0,image_row, window_row):
    				for train_col in range(0,image_col, window_col):
        				window = image_binary[train_row:train_row+window_row,train_col:train_col+window_col]
					
					train_window_white_pixels=cv2.countNonZero(window)
					train_density=float(train_window_white_pixels)/(window_row*window_col)
			
					train_density_list.append(train_density)
					#print(train_density_list)
					sum=sum+train_window_white_pixels
					train_zones=train_zones+1

			train_density_list=np.float32(train_density_list)
			train_density_array=np.asarray(train_density_list)
			#print(train_density_array)
			#print("Total number of zones %d" %train_zones)
			#print("Sum of white pixels in window %d " %sum)
			#print("Row size density_list %d" %train_density_array.size)


			training_list.append(train_density_array)
			training_labels.append(train_file_name)


#print(labels)
#print("Count of train images %d" %count_of_train_images)
training_matrix = np.asarray(training_list)
training_matrix=np.float32(training_matrix)
print("Training matrix")
print(training_matrix)
#print(training_matrix.shape)

train_label_matrix=np.asarray(training_labels)
train_label_matrix=np.float32(train_label_matrix)
#print(label_matrix)

knn=cv2.KNearest()
knn.train(training_matrix,train_label_matrix)


count_of_test_images=0
testing_list=[]
testing_labels=[]
k=0


#path to the folder where word to be segmented and written into machine editable form is saved. Change path accordingly.
path_word="/WORD"

for root, dirs, files in os.walk(path_word):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.splitext(fullpath)[1] == '.png':
			base=os.path.basename(f)
			os.path.splitext(base)
			file_name_word=os.path.splitext(base)[0]


			word_image=cv2.imread(fullpath)
			cv2.imshow("word_image",word_image)
			test_image_1=word_image[0:72,0:80]
			test_image_2=word_image[0:72,81:160]
			

			name_first_char=file_name_word.split("*")[0]
			name_second_char=file_name_word.split("*")[1]
			
			#Segmented words will be saved in "Wordscropped" folder
			cv2.imwrite("/WordsCropped/"+name_first_char+".jpg",test_image_1)
			cv2.imwrite("/WordsCropped/"+name_second_char+".jpg",test_image_2)
			


			if(len(file_name_word.split("*"))>2):
				test_image_3=word_image[0:72,160:240]
				name_third_char=file_name_word.split("*")[2]
				cv2.imwrite("/WordsCropped/"+name_third_char+".jpg",test_image_3)	


#Testing path i.e. WordsCropped folder
test_path="/WordsCropped"
window_row=5
window_col=5


for root, dirs, files in os.walk(test_path):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.splitext(fullpath)[1] == '.jpg':
			base=os.path.basename(f)
			os.path.splitext(base)

			test_file_name=os.path.splitext(base)[0]
			test_file_name=test_file_name.split("_")[0]
			
			count_of_test_images=count_of_test_images+1

			test_image = cv2.imread(fullpath)
			cropped_test_image = test_image[5:94, 5:94]
			resized_test_image = cv2.resize(cropped_test_image,(int(60),int(60)))
			test_image_dimensions=resized_test_image.shape		
			test_image_grayscale = cv2.cvtColor(resized_test_image, cv2.COLOR_BGR2GRAY)			
			test_ret,test_image_binary = cv2.threshold(test_image_grayscale,127,255,cv2.THRESH_BINARY)
			
			cv2.imshow('test_image_binary',test_image_binary)

			test_density_list=[]
			test_sum=0
			test_zones=0
			for test_r in range(0,60, window_row):
    				for test_c in range(0,60, window_col):
        				test_window = test_image_binary[test_r:test_r+window_row,test_c:test_c+window_col]
					
					test_window_white_pixels=cv2.countNonZero(test_window)
					
					density_test=float(test_window_white_pixels)/(window_row*window_col)
					
					test_density_list.append(density_test)

					test_sum=test_sum+test_window_white_pixels
					test_zones=test_zones+1

			test_density_list=np.float32(test_density_list)
			test_density_array=np.asarray(test_density_list)

			testing_list.append(test_density_array)
			testing_labels.append([])
			testing_labels[k].append(test_file_name)
			k=k+1
			

print("Test images %d" %count_of_test_images)

print("Test matrix")
testing_matrix = np.asarray(testing_list)
testing_matrix=	np.float32(testing_matrix)
print(testing_matrix)

print("Testing label matrix")
testing_label_matrix=np.asarray(testing_labels)
testing_label_matrix=np.float32(testing_label_matrix)
print(testing_label_matrix)
			

ret,result,neighbours,dist=knn.find_nearest(testing_matrix,k=1)

print("Result of knn")
print(result)

matches = (result==testing_label_matrix)
print(matches.shape)
print(matches)

print("distance")
print(dist)

print("neighbours")
print(neighbours)

correct = np.count_nonzero(matches)
accuracy = (correct*100.0)/result.size
print("Correct %d" %correct)
print("Accuracy %f" %accuracy)
print(accuracy)

unicode={					  	1.0:u'\u0C85',2.0:u'\u0C86',3.0:u'\u0C87',4.0:u'\u0C88',5.0:u'\u0C89',6.0:u'\u0C8A',7.0:u'\u0C8B',8.0:u'\u0C8E',9.0:u'\u0C8F',10.0:u'\u0C90',
11.0:u'\u0C92',12.0:u'\u0C93',13.0:u'\u0C94',14.0:u'\u0C85'u'\u0C82',15.0:u'\u0C83',16.0:u'\u0C95',17.0:u'\u0C96',18.0:u'\u0C97',19.0:u'\u0C98',
20.0:u'\u0C99',21.0:u'\u0C9A',22.0:u'\u0C9B',23.0:u'\u0C9C',24.0:u'\u0C9D',25.0:u'\u0C9E',26.0:u'\u0C9F',27.0:u'\u0CA0',28.0:u'\u0CA1',	29.0:u'\u0CA2',30.0:u'\u0CA3',31.0:u'\u0CA4',32.0:u'\u0CA5',33.0:u'\u0CA6',34.0:u'\u0CA7',35.0:u'\u0CA8',36.0:u'\u0CAA',37.0:u'\u0CAB',
38.0:u'\u0CAC',39.0:u'\u0CAD',40.0:u'\u0CAE',41.0:u'\u0CAF',42.0:u'\u0CB0',43.0:u'\u0CB2',44.0:u'\u0CB5',45.0:u'\u0CB6',46.0:u'\u0CB7',
47.0:u'\u0CB8',48.0:u'\u0CB9'
}


j=0
for j in range(result.size):
	key=result[j,0]
	print(key)
	print(type(key))
	unicode_File_Name.write("%s" %unicode[key])
	j=j+1




cv2.waitKey(0)
cv2.destroyAllWindows()
 
			
