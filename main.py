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
import matplotlib.pyplot as plt
import pyttsx3
#from .engine import Engine
engine =pyttsx3.init()
modelpath='detect.tflite'
lblpath='labelmap.txt'
min_conf=0.5
cap=cv2.VideoCapture(-1,cv2.CAP_V4L)
sys.path.append("/home/muhd/Desktop/TF-OBJ/models/research")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


label_map = label_map_util.load_labelmap(lblpath)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=23, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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
    print("hi")
    def region_of_interest(img, vertices):
        print("entered")
        mask = np.zeros_like(img)   
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255   
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        print("works f9")
        return masked_image

    #video = cv2.VideoCapture(0)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    # out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
    #out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20 , (frame_width,frame_height))
    try:

    
        while(cap.isOpened()):
            print("while condition entered")
            ret, frame = cap.read()
            stime = time.time()
            objects = []
            class_str = ""
            frame_width = frame.shape[0]
            frame_height = frame.shape[1]
            rows, cols = frame.shape[:2]
            left_boundary = [int(cols*0.40), int(rows*0.95)]
            left_boundary_top = [int(cols*0.40), int(rows*0.20)]
            right_boundary = [int(cols*0.60), int(rows*0.95)]
            right_boundary_top = [int(cols*0.60), int(rows*0.20)]
            bottom_left  = [int(cols*0.20), int(rows*0.95)]
            top_left     = [int(cols*0.20), int(rows*0.20)]
            bottom_right = [int(cols*0.80), int(rows*0.95)]
            top_right    = [int(cols*0.80), int(rows*0.20)]
            vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
            cv2.line(frame,tuple(bottom_left),tuple(bottom_right), (255, 0, 0), 5)
            cv2.line(frame,tuple(bottom_right),tuple(top_right), (255, 0, 0), 5)
            cv2.line(frame,tuple(top_left),tuple(bottom_left), (255, 0, 0), 5)
            cv2.line(frame,tuple(top_left),tuple(top_right), (255, 0, 0), 5)
            copied = np.copy(frame)
            interested=region_of_interest(copied,vertices)


            frame_expanded = np.expand_dims(interested, axis=0)
                    # image_np_expanded = np.expand_dims(image_np, axis=0)
            if float_input:
                frame_expanded = (np.float32(frame_expanded) - input_mean) / input_std
                    
            print("ok till here")
            interpreter.set_tensor(input_details[0]['index'], frame_expanded)
            print("input data being read")
            interpreter.invoke()
            # image_tensor = interpreter.get_tensor_by_name('image_tensor:0')
            #boxes = interpreter.get_tensor_by_name('detection_boxes:0')
            #     # Each score represent how level of confidence for each of the objects.
            #     # Score is shown on the result image, together with the class label.
            # scores = interpreter.get_tensor_by_name('detection_scores:0')
            #classes = interpreter.get_tensor_by_name('detection_classes:0')
            # num_detections = int(interpreter.get_tensor_by_name('num_detections:0'))
            # Actual detection.
            

            #print(boxes)
            #print(classes)
            # (boxes, scores, classes, num) = sess.run(
            #     [boxes, scores, classes, num_detections],
            #     feed_dict={image_tensor: frame_expanded})
            output_tensors = [interpreter.get_tensor(output_details['index']) for output_details in interpreter.get_output_details()]
            boxes, scores, classes, num_detections = output_tensors            
            print("code reached")
            print(boxes)
            print(scores)
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.78)
            print(frame_width,frame_height)
            ymin = int((boxes[0][0][0]*frame_width))
            xmin = int((boxes[0][0][1]*frame_height))
            ymax = int((boxes[0][0][2]*frame_width))
            xmax = int((boxes[0][0][3]*frame_height))
            Result = np.array(frame[ymin:ymax,xmin:xmax])

            ymin_str='y min  = %.2f '%(ymin)
            ymax_str='y max  = %.2f '%(ymax)
            xmin_str='x min  = %.2f '%(xmin)
            xmax_str='x max  = %.2f '%(xmax)
            cv2.putText(frame,ymin_str, (50, 50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            cv2.putText(frame,ymax_str, (50, 70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            cv2.putText(frame,xmin_str, (50, 90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            cv2.putText(frame,xmax_str, (50, 110),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            print(scores.max())
            print("left_boundary[0],right_boundary[0] :", left_boundary[0], right_boundary[0])
            print("left_boundary[1],right_boundary[1] :", left_boundary[1], right_boundary[1])
            print("xmin, xmax :", xmin, xmax)
            print("ymin, ymax :", ymin, ymax)
            if scores.max() > 0.78:
                print("inif")
            if(xmin >= left_boundary[0]):
                print("move LEFT - 1st !!!")
                cv2.putText(frame,'Move LEFT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
                engine.say("move left")
                engine.runAndWait()
            elif(xmax <= right_boundary[0]):
                
                
                print("move Right - 2nd !!!")
                cv2.putText(frame,'Move RIGHT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
                engine.say("move right")
                engine.runAndWait()
            elif(xmin <= left_boundary[0] and xmax >= right_boundary[0]):
                print("STOPPPPPP !!!! - 3nd !!!")
                cv2.putText(frame,' STOPPPPPP!!!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
                engine.say("stop")
                engine.runAndWait()
                time.sleep(2)
                cv2.line(frame,tuple(left_boundary),tuple(left_boundary_top), (255, 0, 0), 5)
                cv2.line(frame,tuple(right_boundary),tuple(right_boundary_top), (255, 0, 0), 5)

            print("execution terminated")      
            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except:
        pass
