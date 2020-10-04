import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

#load model the final version is number 17 at 30 epochs and 61% accuracy 
model = model_from_json(open("emotions17.json", "r").read())
#load weights
model.load_weights('emotions17.h5')

# Import CV2 Cascade to detect frontal faces
frontal_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Option to use the webcam 
cap=cv2.VideoCapture(0)

# Option to use a recorded video (but is super slow, way better using the webcam)
#cap = cv2.VideoCapture('test3.mp4') 

# Create a loop to find faces + return the position of the faces + display the emotion with the maximum score.

while True:
    stat,first_image=cap.read()# read the image from the webcam / cap.read() returns a bool (True/False) if read correctly
    if not stat: # if correct continue
        continue
    gray_image= cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY) # transform to a gray image color, it help us identify important edges or other features

    faces_detected = frontal_face_cascade.detectMultiScale(gray_image,1.32, 5) # (gray image, scale factor and min neighbors)

# Find faces and return the position on of the faces
    for (x,y,w,h) in faces_detected: # draw a rectangle around the faces
        cv2.rectangle(first_image,(x,y),(x+w,y+h),(255,0,0), 7) # (start point,end point) + rectangle color + thicknes 
        roi_gray=gray_image[y:y+w,x:x+h] #crop the face area from image (start point,end point)
        roi_gray=cv2.resize(roi_gray,(48,48)) # in the model we use 48x48
        img_pixels = image.img_to_array(roi_gray) # convert image to numpy array
        img_pixels = np.expand_dims(img_pixels, axis = 0) # expand the shape of the array into row multiple columns
        img_pixels = img_pixels / 255  # because pixels are on a scale from 0 to 255 (normalize)

# apply the model to the image in pixels
        predictions = model.predict(img_pixels)

#Find the max score for the emotion (as indexed array) 
        max_index = np.argmax(predictions[0])
        emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')       
        prediction = emotions[max_index]  # return the emotion with the highest score

# Add the emotion text to the face detection rectangle
        cv2.putText(first_image, prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,239), 2)

    resized_image = cv2.resize(first_image, (900, 600)) # if not the image/frame looks wide and boxy , with this it looks better 
    cv2.imshow('Facial Emotion Detection',resized_image) # show + add title to the window

# When 'q' key is pressed quit after 15 seconds

    if cv2.waitKey(15) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
