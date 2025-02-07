import cv2
import os
import uuid
## For taking the verification images for which the model is predicting the image

cap = cv2.VideoCapture(0)         # making object of a video capturing device
while cap.isOpened():
    ret, frame = cap.read()
    
    # cut down frame to 250 x 250 pixels
    frame = frame[170:170+250,200:200+250,:]        # this is standard image slicing we're slicing down the image to 250x250 pixels

    cv2.imshow('Verification_Save', frame)             # name of the camera module whih is opened in front 

    # Verification trigger with key 'v'
    if cv2.waitKey(10) & 0xFF == ord('s'):
        
        # save input image to application_data/input_image folder
        cv2.imwrite(os.path.join('application_data', 'verification_images', '{}.jpg'.format(uuid.uuid1())), frame)        # folder + file name 
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()