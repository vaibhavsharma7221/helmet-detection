# import necessary packages
from imutils.video import VideoStream
import numpy as np
from imutils.video import FPS
import imutils
import time
import cv2
from keras.models import load_model

# initialize the list of class labels MobileNet SSD was trained to detect
# generate a set of bounding box colors for each class
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
#CLASSES = ['motorbike', 'person']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

print('Loading helmet model...')
loaded_model = load_model('new_helmet_model.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# initialize the video stream,
print("[INFO] starting video stream...")

# Loading the video file
cap = cv2.VideoCapture('vid1.mp4') 

# time.sleep(2.0)

# Starting the FPS calculation
fps = FPS().start()

# loop over the frames from the video stream
# i = True
while True:
	# i = not i
	# if i==True:

    try:
        # grab the frame from the threaded video stream and resize it
        # to have a maxm width and height of 600 pixels
