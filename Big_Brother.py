#Importing the libraries
import cv2
import numpy as np
import pandas as pd

#Import cascade classifier which will be used to detect the face
face_cascade=cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt.xml')

#Import the trained Facial Recognition Model created using Model_Train
recognizer= cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

#Import Labels created while training the Recognizer
Labels_df=pd.read_csv("Labels.txt", sep=",", header=0)
Labels=dict(zip(list(Labels_df['Code']),list(Labels_df['Name'])))

#Start webcam and recorder
video=cv2.VideoCapture(0)

#Parameters necessary for creating a Rectangle around the Region of Interest and to print name of the individual recognised
color= (255,0,0)   #BGR instead of the usual RGB
stroke= 2
font=cv2.FONT_HERSHEY_SIMPLEX

while True:
    check, frame=video.read()                               #read the current frame of the video
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)       #convert the frame seen in 15 to 'grayScale'
    gray_frame=cv2.GaussianBlur(gray_frame,(5,5),0)         #GaussianBlur helps supress the noise by using a Gaussian kernel. Refer "https://computergraphics.stackexchange.com/questions/39/how-is-gaussian-blur-implemented"
    
    #Use the Face Cascade to get the co-ordinates and dimensions of all the faces in the image
    faces= face_cascade.detectMultiScale(gray_frame,scaleFactor=1.5,minNeighbors=5)
    
    #recognize the face in the video
    for (x,y,w,h) in faces:
        roi_gray= gray_frame[y:y+h,x:x+w]                       #get the numpy array of the region of interest
        id_, confidence= recognizer.predict(roi_gray)           #use the Facial Recognition Model to predict the face
        
        #Keep the prediction of there is a 50% match.
        if confidence>=50:
            #print(confidence, "    ", Labels[id_])                                         #print out the confidence of the prediction and the name of the person in the console.
            cv2.putText(frame, Labels[id_], (x,y), font, 1, color, stroke, cv2.LINE_AA)     #Display name of the individual in the frame.
            cv2.imwrite(Labels[id_]+".png", roi_gray)                                       #Save an image of the face detected with the name.
            
            #Draw a rectangle around the RoI.
            cv2.rectangle(frame,(x,y),(x+w,y+h), color, stroke)
            cv2.rectangle(gray_frame,(x,y),(x+w,y+h), color, stroke)
            
        #else:
            #print(confidence, "    ", Labels[id_])             #To know the closest prediction and its confidence.
            
    #Create Windows to display all the frames
    cv2.imshow("Gray Frame",gray_frame)
    cv2.imshow("Color Frame",frame)
    
    #Command to quit the application by pressing "q" or "Q"
    key=cv2.waitKey(1)
    if key==ord("q") or key==ord("Q"):
        break

#Stop recording and close the webcam windows
video.release()
cv2.destroyAllWindows()

#################### CODE ENDS ####################