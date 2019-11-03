# Big_Brother
A facial recognition webcam application. Uses a set of input data to train a recognizer and then uses it to detect the person in the video.

HOW IT WORKS?
1. Uses the OpenCV Library's LBPHFaceRecognizer and to train the model on images saved in the "Known_Faces" folder.
2. The model is saved and accessed by the master file.
3. The master file uses the haarcascade_frontalface_alt to detect faces. These faces are then passed to the Recognizer model, which displays the name of the person in the frame.
4. The application saves the face of all the individuals who entered the frame along with their name. 

LIMITATION:
The recognizer has worked with high precision but the face detection has not been completely accurate. Working on finding a better model.
