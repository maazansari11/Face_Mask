#!/usr/bin/python
import socket
from datetime import datetime

import cv2
import numpy
import random
import sys
import time
host = "localhost"#sys.argv[1] # e.g. localhost, 192.168.1.123
cam_url = "rtspsrc location=rtsp://admin:India12345@195.229.90.110:4444 latency=2000 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink" #sys.argv[2] # rtsp://user:pass@url/live.sdp , http://url/video.mjpg ...
cam_url = "rtsp://admin:India12345@195.229.90.110:4444/H264?ch=1&subtype=0"
cam_url = "0~0"
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect(("192.168.10.105", 5005))
name = str(random.random()) # gives random name to create window

client_socket.send(str.encode(cam_url))

def rcv():
    data = b''
    while 1:
        #print("test1")
        try:
            #print("start1")
            r = client_socket.recv(90456)
            #print(r)
            #print("start2")
            if len(r) == 0:
                exit(0)
            a = r.find(b'END!')
            if a != -1:
                print(r);
                data += r[:a]
                break
            data += r
        except Exception as e:
            print(e)
            continue
    #print(data)
    nparr = numpy.fromstring(data, numpy.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if type(frame) is type(None):
        pass
    else:
        try:

            frame = cv2.resize(frame,(400,400))
            time = datetime.now()
            filename = "images\\frame_{time}.jpg".format(time=time.strftime("%Y_%m_%d_%H_%M_%S_%f"))

            cv2.imwrite(filename,frame);
            cv2.imshow(name,frame)
            if cv2.waitKey(10) == ord('q'):
                client_socket.close()
                sys.exit()
        except:
            client_socket.close()
            exit(0)

while 1:
    print("test")
    rcv()
