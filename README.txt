README

1.Installation of opencv in ubuntu

OpenCV-Python can be installed Install package python-opencv with following command in terminal (as root user).
$ sudo apt-get install python-opencv

Reference : https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html

-----------------------------------------------------------------------------------------------

2.Installation of opencv in Windows
Installing OpenCV from prebuilt binaries

Below Python packages are to be downloaded and installed to their default locations.
Python-2.7.x.
Numpy.
Matplotlib 

Install all packages into their default locations. Python will be installed to C:/Python27/.
After installation, open Python IDLE. Enter import numpy and make sure Numpy is working fine.
Download latest OpenCV release from sourceforge site and double-click to extract it.
Goto opencv/build/python/2.7 folder.
Copy cv2.pyd to C:/Python27/lib/site-packages.

Reference : https://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html

-----------------------------------------------------------------------------------------------

3.Codes

a) knnMomentsAccuracyFinal.py

Execute : python knnMomentsAccuracyFianl.py in terminal
Feature extraction method : Hu's invariant moments
Classification method:	kNN
Output : Accuracy of characters recognized correctly

-----------------------------------------------------

b) SVMMomentsAccuracyFinal.py

Execute : python SVMMomentsAccuracyFinal.py in terminal
Feature extraction method : Hu's invariant moments
Classification method:	SVM
Output : Accuracy of characters recognized correctly

------------------------------------------------------

c)knnZoningAccuracyFinal.py

Execute : python knnZoningAccuracyFinal.py in terminal
Feature extraction method : Zoning
Classification method:	kNN
Output : Accuracy of characters recognized correctly

----------------------------------------------------------
d)SVMZoningAccuracyFinal.py

Execute : python SVMZoningAccuracyFinal.py in terminal
Feature extraction method : Zoning
Classification method:	SVM
Output : Accuracy of characters recognized correctly

------------------------------------------------------------

e)knnZoningWordsFinal.py

Execute : python knnZoningWordsFinal.py in terminal

In this program testing dataset would be a single word given in WORD folder.The word in the "WORD" folder gets cropped into individual letters in WordsCropped Folder.

This program writes the word into Unicode_machine_editable.txt in the Codes folder.

Feature extraction method : Zoning
Classification method:	kNN

Output : Accuracy of characters recognized correctly and word written in Unicode_machine_editable.txt file.

---------------------------------------------------------------

f)segmentation_bounding_box.py

Execute : python segmentation_bounding_box.py in terminal

In this program the input image consisting of word is segmented bases on the modified bounding box method.
The program traverses each pixel column-wise And it will find the connected components.It segments the letters based on the pixel values relative position.
The letters cropped will be written to the path given in the 'variable_name = name' in the code segmentation_bounding_box.py
Any connected component method works on the concept of finding all the nearby connected pixel values.If the letters are more overlapped it will segment them into one letter.

Output: cropped letters



