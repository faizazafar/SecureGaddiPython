import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils import visualization_utils as vis_util
import datetime
from plot import sms
from utils import backbone
import time

import urllib
import requests

from time import mktime

from threading import Thread

import threading
import shutil
import os
import urllib
import requests
import json
import math


def getcoor():
    url = "http://18.224.158.253/plac/public/api/upload-video";
    global refPt

    url = "http://18.224.158.253/plac/public/api/get-coordinates?user_id=1&camera_id=1";
    # url="http://192.168.18.225/plac/public/api/get-coordinates?user_id=1&camera_id=1&timestamp=2021-08-28 04:10:00";

    ts = time.time()
    # dt  = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d,%H:%M:%S')
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    #print(dt)
    user_ID = '1'
    current_file = 'car3.mp4'
    vidfiles = {'video': (current_file, open(current_file, 'rb'))}
    datas = {'user_id': user_ID, 'camera_id': '1', 'date_time': str(dt)}
    #print(datas)
    rsp = requests.get(url)

    json_response = rsp.json()



    sum=0
    for item in range(len(json_response['data'])):
        #print(json_response['data'][item])
        sum=sum+1

    print(sum)

    json_response2 = json_response['data'][sum-1];

    #print(len(json_response))
    print(json_response2['realtime_coord'])
    x_value = (json_response2['realtime_coord']['x_value']);
    y_value = (json_response2['realtime_coord']['y_value']);
    x_value2 = (json_response2['realtime_coord']['x2_value']);
    y_value2 = (json_response2['realtime_coord']['y2_value']);
    x_value3 = (json_response2['realtime_coord']['x3_value']);
    y_value3 = (json_response2['realtime_coord']['y3_value']);
    x_value4 = (json_response2['realtime_coord']['x4_value']);
    y_value4 = (json_response2['realtime_coord']['y4_value']);
    print(x_value)
    str2=x_value.split("'")
    str3=y_value.split("'")

    # x1 =float(str2[0])
    # y1 =float(str3[0])
    # x1=math.floor(-350+x1*9)
    # y1=math.floor(-600+y1*3)
    # x2 = math.floor(-350+x_value4*9)
    # y2 = math.floor(-600+y_value4*3)
    #
    x1 = float(str2[0])
    y1 =float(str3[0])
    x1=math.floor(100+x1*1.3)
    y1=math.floor(-320+y1*1.5)
    x2 = math.floor(100+x_value4*1.3)
    y2 = math.floor(-320+y_value4*1.5)

    # x1 = float(str2[0])
    # y1 = float(str3[0])
    # x1 = math.floor(x1)
    # y1 = math.floor(y1)
    # x2 = math.floor(x_value4)
    # y2 = math.floor(y_value4)

    refPt = [(x1, y1)]
    #refPt.append((x_value4, y_value4))
    #refPt.append((x_value2, y_value2))
    #refPt.append((x_value3, y_value3))
    refPt.append((x2, y2))


    cv2.rectangle(first_frame, (x1,y1),(x2,y2), (0, 255, 0), 2)
    cv2.imshow("image", first_frame)
    print(refPt)


    x_value=78
    y_value=279
    x_value4=156
    y_value6=209
    #cv2.rectangle(first_frame, (strx_value,y_value),(x_value4,y_value4), (0, 255, 0), 2)
    #cv2.imshow("image", first_frame)
def remove_finalvideo(cap,Truck):
      workingdirectory = os.getcwd()
      outDirectory2 = workingdirectory + '//outputStreamFinal//'
      files = os.listdir(outDirectory2)
      print(len(files))
      current_file = ""
      sum = 0;
      newsum = 0
      oldest_file = sorted([outDirectory2 + "" + f for f in files], key=os.path.getctime)

      for val in oldest_file:
       print("removing")

       path_dir_in_parts = oldest_file[newsum].split("//");
       #print(path_dir_in_parts)

       current_file = path_dir_in_parts[2]
       print(outDirectory2 + current_file)
       os.remove(outDirectory2+current_file)
       newsum = newsum + 1
      # if (rsp.json()['success'] == True):
      # print(rsp.text)

def upload_finalvideo(cap,Truck) :


  url2 = "http://18.224.158.253/plac/public/api/upload-video";
  url = "http://18.224.158.253/plac/public/api/upload-alert";

  workingdirectory = os.getcwd()
  outDirectory2 = workingdirectory + '//outputStreamFinal//'
  files = os.listdir(outDirectory2)
  print(len(files))
  current_file = ""
  sum=0;
  newsum=0
  oldest_file = sorted([outDirectory2 + "" + f for f in files], key=os.path.getctime)
  for val in oldest_file:
    print ("sum")
    print(sum)
    print(oldest_file[sum])

    path_dir_in_parts = oldest_file[sum].split("//");
    print(path_dir_in_parts)
    current_file = path_dir_in_parts[2]
    sum = sum + 1
 #if (len(files) > 0):
    # lastest_file =files[len(files)-1]
  #  if (len(files) > 1):
   #     oldest_file = sorted([outDirectory2 + "" + f for f in files], key=os.path.getctime)
    #    # print( oldest_file[0] )
     #   path_dir_in_parts = oldest_file[0].split("//");
     #   print(path_dir_in_parts)
     #   current_file = path_dir_in_parts[2]
     #   # print(path_dir_in_parts[6])
   # else:
    #    current_file = files[0]
     #   # print(files)
    print(current_file)
    timeFile = current_file.split(".")
    dt = timeFile[0].split("_")
    dt[1] = dt[1].replace('-', ':')
    dt = " ".join(dt)
    # print(dt)

    #ts = time.time() + 300
# dt  = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d,%H:%M:%S')
# ts = datetime.now()
# dt = time.mktime(ts.timetuple()).strftime('%Y-%m-%d %H:%M:%S')
    ts=time.time()+600
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(dt)
# date_to_strp = time.strptime(ts, '%Y%m%d%H%M%S')
# date_final = datetime.fromtimestamp(mktime(date_to_strp))

# print(dt)
    user_ID = '1'
    movement_type="Car Move"
    remarks="Suspicious"
    is_suspected=1;
# current_file = 'car3.mp4'
# vidfiles = {'video': (current_file,open(current_file, 'rb'))}
    #datas = {'user_id': user_ID, 'camera_id': '1', 'date_time': str(dt)}

    datas_alert = {'user_id' :  user_ID  , 'camera_id' : '1', 'date_time' : str(dt), 'movement_type' : str(movement_type), 'remarks' : str(remarks), 'is_suspected' : str(is_suspected) }
# print(datas)
    image = "youtube.jpg"
# files = {'media': open('youtube.jpg', 'rb')}

    current_file2 = outDirectory2 + current_file;

    vidfiles = {'video': (current_file2, open(current_file2, 'rb')), 'thumbnail': (image, open(image, 'rb'))}

    rsp = requests.post(url, data=datas_alert, files=vidfiles)
    print("start final uploading")
    print(outDirectory2 + current_file)
    #os.remove(current_file2)





def upload_normalvideo(cap,Truck) :


  url2 = "http://18.224.158.253/plac/public/api/upload-video";
  url = "http://18.224.158.253/plac/public/api/upload-video";

  workingdirectory = os.getcwd()
  outDirectory2 = workingdirectory + '//outputStream//'
  files = os.listdir(outDirectory2)
  print(len(files))
  current_file = ""
  sum=0;
  oldest_file = sorted([outDirectory2 + "" + f for f in files], key=os.path.getctime)
  for val in oldest_file:
    print(oldest_file[sum])
    sum=sum+1
    path_dir_in_parts = oldest_file[sum].split("//");
    print(path_dir_in_parts)
    current_file = path_dir_in_parts[2]

 #if (len(files) > 0):
    # lastest_file =files[len(files)-1]
  #  if (len(files) > 1):
   #     oldest_file = sorted([outDirectory2 + "" + f for f in files], key=os.path.getctime)
    #    # print( oldest_file[0] )
     #   path_dir_in_parts = oldest_file[0].split("//");
     #   print(path_dir_in_parts)
     #   current_file = path_dir_in_parts[2]
     #   # print(path_dir_in_parts[6])
   # else:
    #    current_file = files[0]
     #   # print(files)
    print(current_file)
    timeFile = current_file.split(".")
    dt = timeFile[0].split("_")
    dt[1] = dt[1].replace('-', ':')
    dt = " ".join(dt)
    # print(dt)

    #ts = time.time() + 300
# dt  = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d,%H:%M:%S')
# ts = datetime.now()
# dt = time.mktime(ts.timetuple()).strftime('%Y-%m-%d %H:%M:%S')
    ts=time.time()+600
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(dt)
# date_to_strp = time.strptime(ts, '%Y%m%d%H%M%S')
# date_final = datetime.fromtimestamp(mktime(date_to_strp))

# print(dt)
    user_ID = '1'
# current_file = 'car3.mp4'
# vidfiles = {'video': (current_file,open(current_file, 'rb'))}
    datas = {'user_id': user_ID, 'camera_id': '1', 'date_time': str(dt)}

# datas_alert = {'user_id' :  user_ID  , 'camera_id' : '1', 'date_time' : str(dt), 'movement_type' : str(movement_type), 'remarks' : str(remarks), 'is_suspected' : str(is_suspected) }
# print(datas)
    image = "youtube.jpg"
# files = {'media': open('youtube.jpg', 'rb')}

    current_file2: str = outDirectory2 + current_file;

    vidfiles = {'video': (current_file2, open(current_file2, 'rb')), 'thumbnail': (image, open(image, 'rb'))}

    rsp = requests.post(url, data=datas, files=vidfiles)
    print("start uploading")
    print (outDirectory2 + current_file)

    #if (rsp.json()['success'] == True):
     #print(rsp.text)



def FinalVideoSaved(frame):


   print("frmae No")
   print(frameNo2)
   try:


      print(" ---- Start Final Video Saved ---- ")


      # out = cv2.VideoWriter(outputfolder,0x6134706d, 10.0 , (int(cap.get(3)),int(cap.get(4))),True) #this generate all codecs for any format
      quiteloop = False


      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      dtDisplay = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
      # image,text,origion,font,fontscale,color,thickness,linetype,bottomleftorigion
      #cv2.rectangle(gray, (5, 55), (200, 80), (0, 0, 0), -1)


      #cv2.putText(gray, str(dtDisplay), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
      #cv2.LINE_AA)
      # cv2.line(gray,(5,55),(200,80),(0,0,0),cv2.FILLED)
      cv2.rectangle(gray, (195, 460), (395, 490), (0, 0, 0), cv2.FILLED)
      cv2.putText(gray, str(dtDisplay), (200, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                  cv2.LINE_AA)
      out.write(gray)

      # cv2.imshow('gray', gray)





      # VideoUpload(videoFile,outputfolder,dt)

      #if (quiteloop):
      # break



   except KeyboardInterrupt:
            # webbrowser.open(map_link)        #open current position information in google map
            # sys.exit(0)
            print("Keyboard Exception")

   except Exception as err:
            # webbrowser.open()        #open current position information in google map
            # sys.exit(0)
            error_file = open("errorLog_file.txt", "a")
            error_file.write(
                str(err) + "\n Video Capturing exception " + datetime.datetime.fromtimestamp(time.time()).strftime(
                    '%Y-%m-%d %H:%M:%S'))
            error_file.close()
            # print(sys.exc_info())

            print("Exception")

def VideoCapturing(cap, Truck):
    global frameNo
    try:
        while (cap.isOpened()):
            print(" ---- Start New Video ---- ")
            frameNo = 0
            ts = time.time()
            #dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            videoFile = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S') + '.mp4'
            workingdirectory=os.getcwd()
            outDirectory1 = workingdirectory + '/OutPutData/'
            outputfolder = outDirectory1 + videoFile
           # out = cv2.VideoWriter(outputfolder, cv2.VideoWriter_fourcc(*"avc1"), 10.0,
            #                      (int(cap.get(3)), int(cap.get(4))), False)

            out = cv2.VideoWriter(outputfolder, cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                                  (int(cap.get(3)), int(cap.get(4))), False)
            # out = cv2.VideoWriter(outputfolder,0x6134706d, 10.0 , (int(cap.get(3)),int(cap.get(4))),True) #this generate all codecs for any format
            quiteloop = False

            while (frameNo < 300):
                # capturre frame by frame
                frameNo = frameNo + 1
                ret, frame = cap.read()
                if not ret:
                    break

                # frame = cv2.flip(frame, -1) # Flip camera vertically
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                dtDisplay = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                # image,text,origion,font,fontscale,color,thickness,linetype,bottomleftorigion
               # cv2.rectangle(gray, (5, 55), (200, 80), (0, 0, 0), cv2.FILLED)
                #cv2.putText(gray, str(dtDisplay), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                      #      cv2.LINE_AA)

                cv2.rectangle(gray, (195, 460), (395, 490), (0, 0, 0), cv2.FILLED)
                cv2.putText(gray, str(dtDisplay), (200, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                # cv2.line(gray,(5,55),(200,80),(0,0,0),cv2.FILLED)

                out.write(gray)
                cv2.imshow('frame', gray)
                # cv2.imshow('gray', gray)

                k = cv2.waitKey(30) & 0xff
                if k == 27:  # press 'ESC' to quit
                    quiteloop = True
                    break

            out.release()
            shutil.move(outputfolder, workingdirectory + '/outputStream/')
            # VideoUpload(videoFile,outputfolder,dt)

            if (quiteloop):
                break

        cap.release()
        # out.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        # webbrowser.open(map_link)        #open current position information in google map
        # sys.exit(0)
        print("Keyboard Exception")

    except Exception as err:
        # webbrowser.open()        #open current position information in google map
        # sys.exit(0)
        error_file = open("errorLog_file.txt", "a")
        error_file.write(str(err) + "\n Video Capturing exception " + datetime.datetime.fromtimestamp(time.time()).strftime(
            '%Y-%m-%d %H:%M:%S'))
        error_file.close()
        # print(sys.exc_info())

        print("Exception")
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
     #global refPt
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
     if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
     #refPt2 = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
     elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
         refPt.append((x, y))

         cropping = False
       # draw a rectangle around the region of interest
         cv2.rectangle(first_frame, refPt[0], refPt[1], (0, 255, 0), 2)
         cv2.imshow("image", first_frame)
     print(refPt[0])
     print(refPt[1])


print("start pro")

# cap = cv2.VideoCapture("newcar.mp4")
# cap = cv2.VideoCapture("uitvideo2.mp4")
#cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap=cv2.VideoCapture(0)
ret, first_frame = cap.read()
cv2.namedWindow("image")
#cv2.setMouseCallback("image", click_and_crop)


cv2.waitKey(2)
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')


is_color_recognition_enabled = 1

roi = 0
deviation =2
cnt = 0

global xvalue, yvalue;
xvalue = list(range(0, 300))
yvalue = [0] * 300;
lastsms = [0] * 300;

x_value=0
y_value=0
x_value2=0
y_value2=0
x_value3=0
y_value3=0
x_value4=0
y_value4=0






#CamUploadThread = Thread(target=VideoCapturing, args=(cap, 'Truck 0'))

#CamUploadThread.start()
#VideoUploadThread = Thread(target=upload_normalvideo,args=(cap,'Truck 0'))
#VideoUploadThread.start()

#VideoUploadFinalThread = Thread(target=upload_finalvideo,args=(cap,'Truck 0'))
#VideoUploadFinalThread.start()
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

    #cap=cv2.VideoCapture(0)
    #if cv2.VideoCapture(0).isOpened():
    #    cv2.VideoCapture(0).release()

    #cap1 = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT)




if cv2.waitKey(2) & 0xFF == ord('s'):
	cv2.destroyAllWindows()



def check_car():
 getcoor()

 with detection_graph.as_default():
   with tf.compat.v1.Session(graph=detection_graph) as sess:
     image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
     detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
     detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
     detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
     num_detections = detection_graph.get_tensor_by_name('num_detections:0')
     f = open("demofile3.txt", "a")
     lastframe =0 ;
     nums=0;
     print("detection start")
     lastsmstime=0;
     firsttime_cardetection = 0;
     while (cap.isOpened()):
        ret, frame = cap.read()

        if(nums==0):
          firsttime = datetime.datetime.now()
          nums=nums+1;
        input_frame = frame
        x = refPt[0][0]
        y = refPt[0][1]
        h = refPt[1][1] - refPt[0][1]
        w = refPt[1][0] - refPt[0][0]

        crop_frame = input_frame[y:y + h, x:x + w]

        image_np_expanded = np.expand_dims(crop_frame, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
         feed_dict={image_tensor: image_np_expanded})
        counter, csv_line, counting_mode, min_score_thresh = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(
            cap.get(1),
            input_frame,
            2,
            is_color_recognition_enabled,
            np.squeeze(
                boxes),
            np.squeeze(
                classes).astype(
                np.int32),
            np.squeeze(
                scores),
            category_index,
            y_reference=roi,
            min_score_thresh=.5,
            deviation=deviation,
            use_normalized_coordinates=True,
            line_thickness=4)
        flood=0;
        n=0;

        typeofdetection=[0]*(int(num[0]))
        for i in range(0,int(num[0])):
            typeofdetection[i]=int(classes[0][i])

        for i in range(0,int(num[0])) :
         if ((typeofdetection[i])==3 and scores[0][i]>min_score_thresh):
            #print('car detected')
            firsttime_cardetection=1;

            flood=1;
            Startmult = datetime.datetime.now()
            diff = Startmult - firsttime
            a = diff.total_seconds()
            lastsmstime=a

            #if ((diff.total_seconds()-lastframe)>20):
             #   lastframe=diff.total_seconds()
              #  f.write(str(diff.total_seconds()))
               # f.write("\n")
               # print (str(diff.total_seconds()))
               # index=int(diff.total_seconds())
                #yvalue[index] =1

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                input_frame,
                'Vehicle Detected ',
                (10, 70),
                font,
                0.8,
                (0, 0xFF, 0xFF),
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )

        else:
            font = cv2.FONT_HERSHEY_SIMPLEX

            Startmult = datetime.datetime.now()
            diff = Startmult - firsttime

            index = int(diff.total_seconds())

            if ((index - lastsmstime) > 20 and firsttime_cardetection==1):
                lastsms[index] = 1
                lastsmstime = index
                print("car has moved")
                call_alert(input_frame)
                #sms()
        # start_point = refPt[0]
        # end_point = refPt[1]
        # color = (255, 0, 0)
        # thickness = 2
        # flood=0
        # #if (flood==1):
        #  #           image = cv2.rectangle(input_frame, start_point, end_point, color, thickness)
        #
        # #elif (flood==0):
        # showdetectedarea="Area of Interest"
        # image = cv2.rectangle(input_frame, start_point, end_point, color, +4)
        # cv2.rectangle(image, (refPt[0][0], refPt[0][1]), (refPt[0][0]+50, refPt[0][1]), (0, 0, 0), cv2.FILLED)
        # cv2.putText(image, str(showdetectedarea), (refPt[0][0], refPt[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        #             cv2.LINE_AA)
        #
        # #cv2.imshow('object counting', image)
        # global frameNo2
        # frameNo2 = 0;
        # global out
        # ts = time.time()
        # # dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        # videoFile = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S') + '.mp4'
        # workingdirectory = os.getcwd()
        # outDirectory1 = workingdirectory + '/OutPutDataFinal/'
        # outputfolder = outDirectory1 + videoFile
        # # out = cv2.VideoWriter(outputfolder, cv2.VideoWriter_fourcc(*"avc1"), 10.0,
        # #                      (int(cap.get(3)), int(cap.get(4))), False)
        # out = cv2.VideoWriter(outputfolder, cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
        #                       (int(cap.get(3)), int(cap.get(4))), False)
        #
        # while (frameNo2 < 3000):
        #     # capturre frame by frame
        #     FinalVideoSaved(input_frame)
        #     frameNo2 = frameNo2 + 1
        #
        # out.release()
        # shutil.move(outputfolder, workingdirectory + '/outputStreamFinal/')
        cv2.imshow('live video',input_frame)

        if ret == True:
            crop_frame = frame[y:y + h, x:x + w]

          #  cv2.imshow('croped', crop_frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break



     cap.release()

   f.close()





def call_alert(input_frame):
    start_point = refPt[0]
    end_point = refPt[1]
    color = (255, 0, 0)
    thickness = 2
    flood = 0
    # if (flood==1):
    #           image = cv2.rectangle(input_frame, start_point, end_point, color, thickness)

    # elif (flood==0):
    showdetectedarea = "Area of Interest"
    image = cv2.rectangle(input_frame, start_point, end_point, color, +4)
    cv2.rectangle(image, (refPt[0][0], refPt[0][1]), (refPt[0][0] + 50, refPt[0][1]), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, str(showdetectedarea), (refPt[0][0], refPt[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1,
                cv2.LINE_AA)

    # cv2.imshow('object counting', image)
    global frameNo2
    frameNo2 = 0;
    global out
    ts = time.time()
    # dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    videoFile = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S') + '.mp4'
    workingdirectory = os.getcwd()
    outDirectory1 = workingdirectory + '/OutPutDataFinal/'
    outputfolder = outDirectory1 + videoFile
    # out = cv2.VideoWriter(outputfolder, cv2.VideoWriter_fourcc(*"avc1"), 10.0,
    #                      (int(cap.get(3)), int(cap.get(4))), False)
    out = cv2.VideoWriter(outputfolder, cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                          (int(cap.get(3)), int(cap.get(4))), False)

    while (frameNo2 < 300):
        # capturre frame by frame
        FinalVideoSaved(input_frame)
        frameNo2 = frameNo2 + 1

    out.release()
    shutil.move(outputfolder, workingdirectory + '/outputStreamFinal/')


def save_camera():
 cap = cv2.VideoCapture(0)
 print("success camera")
 cap.set(3, 480)  # set Width
 cap.set(4, 380)  # set Height

 dt = str(datetime.datetime.now())
  # tme = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f" )
 dt_time = dt.split('.')

 outputfolder = "OutPutData/"+ str(dt_time[0]) +'.avi'
 out = cv2.VideoWriter(outputfolder, cv2.VideoWriter_fourcc(*"XVID"), 20.0 , (int(cap.get(3)),int(cap.get(4))))

if cv2.waitKey(0) & 0xFF == ord('u'):
    upload_normalvideo(cap ,'truck 0')
if cv2.waitKey(0) & 0xFF == ord('t'):
    print ("check car start")
    check_car()
if cv2.waitKey(0) & 0xFF == ord('g'):
    call_alert()

if cv2.waitKey(0) & 0xFF == ord('c'):
    getcoor()
if cv2.waitKey(0) & 0xFF == ord('l'):
    upload_finalvideo(first_frame,'truck 0')
if cv2.waitKey(0) & 0xFF == ord('r'):
    remove_finalvideo(first_frame, 'truck 0')


if cv2.waitKey(0) & 0xFF == ord('s'):
        print(xvalue)
        print(yvalue)
        plt.bar(xvalue, lastsms)
        plt.show()

