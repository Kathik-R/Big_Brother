#Importing the libraries
import os
from glob import glob
import numpy as np
import cv2
import pandas as pd

#To define all the directories that will be used in the code.
base_dir= os.path.dirname(os.path.abspath(__file__))
image_dir= os.path.join(base_dir, "Known_Faces")

#Import cascade classifier which will be used to detect the face.
face_cascade=cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt.xml')

#Import the LBPHFaceRecognizer which will be used to create the recognition model using the training set.
recognizer= cv2.face.LBPHFaceRecognizer_create()

#Empty train data. It will contain numpy arrays of the training images (x_train) and an integer representing the individual in the picture (y_train)
x_train=[]
y_train=[]

#Parameters to be used in creating a number-to-label mapping for the people in the training data. 
#Each individual in the training data will have an integer as a representative in y_train.
#This number-to-label mapping will be used in the master file to get the name of the person represented by the integer output of the face recognizer.
label_as_number=0
label_dict={'Name':'Code'}

for folder in glob(image_dir+"/*/"):
    #Add the individual's name, as key, and a number to represent him, as value, to the Label_Dictionary
    label=folder.replace(image_dir,"").replace("\\","")
    label_dict[label]=label_as_number
    label_as_number+=1
    
    for file in os.listdir(folder):
        path= folder+file
        image= cv2.imread(path)                                                                         #Read all the files in the folder.
        image_gray_initial= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)                                      #Convert the image to grayscale
        image_gray_resized=cv2.resize(image_gray_initial,(550,550),interpolation=cv2.INTER_AREA)        #Resize the image.
        faces= face_cascade.detectMultiScale(image_gray_resized,scaleFactor=1.5,minNeighbors=5)         #Detect the face in the image and get their coordinates.
        
        #Extract the Region of Interest from the entire image and then append it to the training data (x_train). 
        #Simultaneously append the number representation of the persons's name to y_train.
        for (x,y,w,h) in faces:
            roi= image_gray_resized[y:y+h,x:x+w]
            x_train.append(roi)
            y_train.append(label_dict[label])

#Save the Label Dictionary as a text file.
(pd.DataFrame.from_dict(data=label_dict, orient='index')).to_csv("Labels.txt", header=False)

#Train the Facial Recognition model and save it, to be later used by the master file.
recognizer.train(x_train,np.array(y_train))
recognizer.save("trainer.yml")

#################### CODE ENDS ####################