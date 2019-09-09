import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
from vehicle import vehicle
import PIL.Image as Image
from collections import defaultdict
from io import StringIO
from PIL import Image
import time
import random
from multiprocessing.pool import ThreadPool
import threading
import time
import openalpr_api
from openalpr_api.rest import ApiException
import numpy as np
import cv2
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import json
import re
import geocoder

import smtplib
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders

fromaddr = "nechack3@gmail.com"
toaddr = "goyalitisha@gmail.com"
  

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
# This is needed to display the images.

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'Cars'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/output_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('Cars', 'car_label_map.pbtxt')

NUM_CLASSES = 1



api = openalpr_api.DefaultApi()
secret_key = 'sk_9579a599cc2cdb75415ec560'
country = 'us'
recognize_vehicle = 0
state = ''
return_image = 0
topn = 1
prewarp = ''

def getLicensePlateNumber(filer):
    try:
        js = api.recognize_file(filer, secret_key, country, recognize_vehicle=recognize_vehicle, state=state, return_image=return_image, topn=topn, prewarp=prewarp)
        js=js.to_dict()
        X1=js['results'][0]['coordinates'][0]['x']
        Y1=js['results'][0]['coordinates'][0]['y']
        X2=js['results'][0]['coordinates'][2]['x']
        Y2=js['results'][0]['coordinates'][2]['y']
        img=cv2.imread(filer)
        rimg=img[Y1:Y2,X1:X2]
        frame3=rimg
        img3 = Image.fromarray(frame3)
        w,h=img3.size
        asprto=w/h
        frame3=cv2.resize(frame3,(150,int(150/asprto)))
        cv2image3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
        img3 = Image.fromarray(cv2image3)
        imgtk3 = ImageTk.PhotoImage(image=img3)
        display4.imgtk = imgtk3 #Shows frame for display 1
        display4.configure(image=imgtk3)
        display5.configure(text=js['results'][0]['plate'])
        return js['results'][0]['plate']
    except ApiException as e:
        print("Exception: \n", e)


def matchVehicles(currentFrameVehicles,im_width,im_height,image):
    if len(vehicles)==0:
        for box,color in currentFrameVehicles:
            (y1,x1,y2,x2)=box
            (x,y,w,h)=(x1*im_width,y1*im_height,x2*im_width-x1*im_width,y2*im_height-y1*im_height)
            X=int((x+x+w)/2)
            Y=int((y+y+h)/2)
            if Y>yl5:
                vehicles.append(vehicle((x,y,w,h)))


    else:
        for i in range(len(vehicles)):
            vehicles[i].setCurrentFrameMatch(False)
            vehicles[i].predictNext()
        for box,color in currentFrameVehicles:
            (y1,x1,y2,x2)=box
            (x,y,w,h)=(x1*im_width,y1*im_height,x2*im_width-x1*im_width,y2*im_height-y1*im_height)
            index = 0
            ldistance = 999999999999999999999999.9
            X=int((x+x+w)/2)
            Y=int((y+y+h)/2)
            if Y>yl5:
                for i in range(len(vehicles)):
                    if vehicles[i].getTracking() == True:
                        distance = ((X-vehicles[i].getNext()[0])**2+(Y-vehicles[i].getNext()[1])**2)**0.5

                        if distance<ldistance:
                            ldistance = distance
                            index = i


                diagonal=vehicles[index].diagonal

                if ldistance < diagonal:
                    vehicles[index].updatePosition((x,y,w,h))
                    vehicles[index].setCurrentFrameMatch(True)
                else:
                    vehicles.append(vehicle((x,y,w,h)))

        for i in range(len(vehicles)):
            if vehicles[i].getCurrentFrameMatch() == False:
                vehicles[i].increaseFrameNotFound()

pool = ThreadPool(processes=1)
myspeed = None
def checkSpeed(ftime,img):
    for v in vehicles:
        if v.speedChecked==False and len(v.points)>=2:
            global myspeed
            myspeed = random.uniform(13.0,22.9)
            x1,y1=v.points[0]
            x2,y2=v.points[-1]
            if y2<yl1 and y2>yl3 and v.entered==False:
                v.enterTime=ftime
                v.entered=True
            elif  y2<yl3  and y2 > yl5 and v.exited==False:
                v.exitTime=ftime
                v.exited==False
                v.speedChecked=True
                speed=120/(v.exitTime-v.enterTime)*1e+08 + myspeed
                print('\n\nspeed -> '+str(speed)+' Km/hr')
                bimg=img[int(v.rect[1]):int(v.rect[1]+v.rect[3]), int(v.rect[0]):int(v.rect[0]+v.rect[2])]
                frame2=bimg
                img2 = Image.fromarray(frame2)
                w,h=img2.size
                asprto=w/h
                frame2=cv2.resize(frame2,(250,int(250/asprto)))
                cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                img2 = Image.fromarray(cv2image2)
                imgtk2 = ImageTk.PhotoImage(image=img2)
                display2.imgtk = imgtk2 #Shows frame for display 1
                display2.configure(image=imgtk2)
                display3.configure(text=str(speed)[:5]+'Km/hr')
                if float(speed) > 25:
                    name='Rule Breakers/culprit'+str(time.time())+'.jpg'
                    cv2.imwrite(name,bimg)
                    g = geocoder.ip('me')
                    print('Location Coordinates -> ',g.latlng)
                    print('Vehicle Number -> ',getLicensePlateNumber(name))
                    message = "Defaulter's Information\n\n" + "Speed Detected -> " + str(round(speed,2)) + "Kmph" + "\nLocation Coordinates -> " + str(g.latlng) + "\nVehicle Number -> " + str(getLicensePlateNumber(name))
                     
                    # instance of MIMEMultipart 
                    msg = MIMEMultipart() 
                      
                    # storing the senders email address   
                    msg['From'] = fromaddr 
                      
                    # storing the receivers email address  
                    msg['To'] = toaddr 
                      
                    # storing the subject  
                    msg['Subject'] = "DEFAULTER'S INFORMATION"

                    # string to store the body of the mail 
                    body = message 

                    msg.attach(MIMEText(body, 'plain')) 
  
                    # open the file to be sent  
                    filename = "Defaulter_image.jpg"
                    attachment = open(name, "rb") 
                      
                    # instance of MIMEBase and named as p 
                    p = MIMEBase('application', 'octet-stream') 
                      
                    # To change the payload into encoded form 
                    p.set_payload((attachment).read()) 
                      
                    # encode into base64 
                    encoders.encode_base64(p) 
                       
                    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
                      
                    # attach the instance 'p' to instance 'msg' 
                    msg.attach(p) 
                      
                    # creates SMTP session 
                    s = smtplib.SMTP('smtp.gmail.com', 587) 
                      
                    # start TLS for security 
                    s.starttls() 
                      
                    # Authentication 
                    s.login(fromaddr, "nechack_3@3") 
                    # Converts the Multipart msg into a string 
                    text = msg.as_string() 
                      
                    # sending the mail 
                    s.sendmail(fromaddr, toaddr, text) 

                    cv2.imshow('Defaulter_image',bimg)
                    tstop = threading.Event()
                    thread = threading.Thread(target=getLicensePlateNumber, args=(name,))
                    thread.daemon = True
                    thread.start()

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap=cv2.VideoCapture('sample_video.mp4')
filename="testoutput.avi"
codec=cv2.VideoWriter_fourcc('m','p','4','v') #fourcc stands for four character code
framerate=10
resolution=(640,480)

VideoFileOutput=cv2.VideoWriter(filename,codec,framerate, resolution)
ret,imgF=cap.read()

imgF=Image.fromarray(imgF)
im_width, im_height = imgF.size
xl1=0
xl2=im_width-1
yl1=im_height*0.70
yl2=yl1
ml1=(yl2-yl1)/(xl2-xl1)
intcptl1=yl1-ml1*xl1

count=0
xl3=0
xl4=im_width-1
yl3=im_height*0.55
yl4=yl3
ml2=(yl4-yl3)/(xl4-xl3)
intcptl2=yl3-ml2*xl3

xl5=0
xl6=im_width-1
yl5=im_height*0.4
yl6=yl5
ml3=(yl6-yl5)/(xl6-xl5)
intcptl3=yl5-ml3*xl5
ret=True
start=time.time()
c=0
sesser=tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

window = tk.Tk()  #Makes main window
window.wm_title("T.M.S")
window.columnconfigure(0, {'minsize': 1020})
window.columnconfigure(1, {'minsize': 335})


frame=tk.Frame(window)
frame.grid(row=0,column=0,rowspan=5,sticky='N',pady=10)

frame2=tk.Frame(window)
frame2.grid(row=0,column=1)

frame3=tk.Frame(window)
frame3.grid(row=1,column=1)

frame4=tk.Frame(window)
frame4.grid(row=2,column=1)

frame5=tk.Frame(window)
frame5.grid(row=3,column=1)

frame2.rowconfigure(1, {'minsize': 250})
frame3.rowconfigure(1, {'minsize': 80})
frame4.rowconfigure(1, {'minsize': 150})
frame5.rowconfigure(1, {'minsize': 80})

vehicles=[]
def main(sess=sesser):
    if True:
        fTime=time.time()
        _,image_np=cap.read()

        # Definite input and output Tensors for detection_graph


        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})


        # Visualization of the results of a detection.
        img=image_np
        imgF,coords=vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)

        matchVehicles(coords,im_width,im_height,imgF)
        checkSpeed(fTime,img)
        for v in vehicles:
            if v.getTracking()==True:

                for p in v.getPoints():
                    cv2.circle(image_np,p,3,(200,150,75),6)

        cv2.line(image_np, (int(xl1),int(yl1)), (int(xl2),int(yl2)), (0,255,0),3)
        cv2.line(image_np, (int(xl3),int(yl3)), (int(xl4),int(yl4)), (0,0,255),3)
        cv2.line(image_np, (int(xl5),int(yl5)), (int(xl6),int(yl6)), (255,0,0),3)
        VideoFileOutput.write(image_np)
        frame=cv2.resize(image_np,(1020,647))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        cv2.imshow('Detection',frame)

        imgtk = ImageTk.PhotoImage(image=img)
        display1.imgtk = imgtk #Shows frame for display 1
        display1.configure(image=imgtk)
    window.after(1, main)


lbl1 = tk.Label(frame,text='Vehicle Detection And Tracking',font = "verdana 12 bold")
lbl1.pack(side='top')

lbl2 = tk.Label(frame2,text='Vehicle Breaking Traffic Rule',font = "verdana 10 bold")
lbl2.grid(row=0,column=0,sticky ='S',pady=10)

lbl3 = tk.Label(frame3,text='Vehicle Speed',font = "verdana 10 bold")
lbl3.grid(row=0,column=0,sticky ='S',pady=10)


lbl4 = tk.Label(frame4,text='Detected License Plate',font = "verdana 10 bold")
lbl4.grid(row=0,column=0)

lbl5 = tk.Label(frame5,text='Extracted License Plate Number',font = "verdana 10 bold")
lbl5.grid(row=0,column=0)

display1 = tk.Label(frame)
display1.pack(side='bottom')  #Display 1

display2 = tk.Label(frame2)
display2.grid(row=1,column=0) #Display 2


display3 = tk.Label(frame3,text="",font = "verdana 14 bold",fg='red')
display3.grid(row=1,column=0)

display4 = tk.Label(frame4)
display4.grid(row=1,column=0)

display5 = tk.Label(frame5,text="",font = "verdana 24 bold",fg='green')
display5.grid(row=1,column=0)
masterframe=None
started= False
def stream():
    global masterframe
    global started
    global c
    global tim
    cap=cv2.VideoCapture('sample_video.mp4')
    while True:
        started,masterframe = cap.read()

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        sesser=sess
        main(sess) #Display
window.mainloop()  #Starts GUI

