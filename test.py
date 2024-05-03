import numpy as np
import cv2
import os
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter
import time
import matplotlib
import pyttsx3
import matplotlib.pyplot as plt
engine =pyttsx3.init()
modelpath='detect.tflite'
lblpath='labelmap.txt'
min_conf=0.5
#cap=cv2.VideoCapture(-1,cv2.CAP_V4L)
cap='/home/muhd/Pictures/Webcam/left.jpg'

interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

while(True):
    #ret, frame =cap.read()
    frame=cv2.imread(cap)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    stime = time.time()
    objects = []
    class_str = ""
    frame_width = image_resized.shape[0]
    frame_height = image_resized.shape[1]
    rows, cols = image_resized.shape[:2]
    left_boundary = [int(cols*0.40), int(rows*0.95)]
    left_boundary_top = [int(cols*0.40), int(rows*0.20)]
    right_boundary = [int(cols*0.60), int(rows*0.95)]
    print(left_boundary[0])
    print(f"right boundry",{right_boundary[0]})
    right_boundary_top = [int(cols*0.60), int(rows*0.20)]
    bottom_left  = [int(cols*0.20), int(rows*0.95)]
    top_left     = [int(cols*0.20), int(rows*0.20)]
    bottom_right = [int(cols*0.80), int(rows*0.95)]
    top_right    = [int(cols*0.80), int(rows*0.20)]


    line1_x = frame_width // 3
    line2_x = 5 * (frame_width // 3)

    # Draw the dividing lines
    cv2.line(frame, (line1_x, 0), (line1_x, height*2), (255, 0, 0), 2)
    cv2.line(frame, (line2_x, 0), (line2_x, height*2), (255, 0, 0), 2)
    print("mememme")


    input_data = np.expand_dims(image_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:

        input_data = (np.float32(input_data) - input_mean) / input_std
        
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    #print("after invoke")
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    #print("after boxes")
    #print(boxes)
    #print(scores)
    detections = []
    
    
    for i in range(len(scores)):
        #print("inside scores")

        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            #print("inside if statement")
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
           #xmaxi=int(min(imW,(boxes[i][3] * imW)))
            print("xmax",xmax)
 
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
            if(object_name=="footpath"):
                engine.say("climb the footpath")
            elif(100<xmin <300):
                print(label)
                print("move R - 1st !!!")
                cv2.putText(image_resized,'Move LEFT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
                engine.say("move left")
                engine.runAndWait()
                time.sleep(5.7)
                print(xmax,xmin,ymax,ymin)
            #elif(xmax <= right_boundary[0]):
            elif(xmin >199):  
                print(label)
                
                print("move Right - 2nd !!!")
                cv2.putText(image_resized,'Move RIGHT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
                engine.say("move right")
                engine.runAndWait()
                time.sleep(5.7)
                print(xmax,xmin,ymax,ymin)
            elif(xmin<100):
                print("STOPPPPPP !!!! - 3nd !!!")
                cv2.putText(image_resized,' STOPPPPPP!!!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
                engine.say("stop")
                engine.runAndWait()
                time.sleep(2)
                cv2.line(image_resized,tuple(left_boundary),tuple(left_boundary_top), (255, 0, 0), 5)
                cv2.line(image_resized,tuple(right_boundary),tuple(right_boundary_top), (255, 0, 0), 5)
                print(xmax,xmin,ymax,ymin)
    
    print(type(frame))
    cv2.imshow('output',frame)
    if((cv2.waitKey(1000) & 0xFF) == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()
    