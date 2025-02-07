## Import Kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

## Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

## Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

## Import other dependecies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


## Build app and layout
class CamApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint = (1,.8))
        self.button = Button(text="verify",on_press = self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist})       # it allows us to load a pretrained 
                                                                                                           #    model into python from a .h5 file

        # Setup Video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)   # triggering of a loop  # runnig update function on given interval (refer documentation)

        return layout
    

    # Run continuosly to get webcam feed
    def update(self, *args):
        
        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[170:170+250,200:200+250,:]

        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame,0).tostring()    # fliping and converting to string
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')   # converting from image to image texture  size=(height, width)
                                                                                # colorformat is 'bgr' which is opencvs format of image
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')        # converting buf to texture which was an opencv image
                                                                               # buffer format is ubyte (refer documentations  (https://kivy.org/doc/stable/api-kivy.graphics.texture.html#module-kivy.graphics.texture))
        self.web_cam.texture = img_texture
        # we're doing here is converting our raw OPENCV iamge array to a texture for rendering. This setting image equal to that texture....
    

    # Load images from file and convert to 100x100px
    def preprocess(self, file_path):
    
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)      # reading the image from the file path of the function provioded path
    
        # Load in the image
        img = tf.io.decode_jpeg(byte_img)         # decoding the image to array form

        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))

        # Scale image to be between 0 and 1
        img = img/255.0

        # Return image
        return img


    # Verification function to verify function
    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.7

        # capture input image from webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[170:170+250,200:200+250,:]
        cv2.imwrite(SAVE_PATH, frame)                             # writing the image to the saved path........

        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):         # looping through every single image in verification_image directory
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))  # file path for input image which is empty riaghtnow 
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Make predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))          # predicting from the model
            results.append(result)


        # Detection Threshold:-- Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)      # how many of out prediction results are surpassing this detection_threshold

        # Verification Threshold:-- Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))   # divide the detection which are above threshold to number of examples 
        verified = verification > verification_threshold            # True or false based on the conditions

        # Set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'               # if else form 

        # Log out details
        Logger.info(results)
        Logger.info(np.sum(np.array(results)> 0.2))
        Logger.info(np.sum(np.array(results)> 0.3))
        Logger.info(np.sum(np.array(results)> 0.4))
        Logger.info(np.sum(np.array(results)> 0.5))
        Logger.info(np.sum(np.array(results)> 0.6))
        Logger.info(np.sum(np.array(results)> 0.7))
        Logger.info(verified)
        Logger.info(len(os.listdir(os.path.join('application_data', 'verification_images'))))
        Logger.info(verification)

        return results, verified






if __name__=='__main__':
    CamApp().run()

