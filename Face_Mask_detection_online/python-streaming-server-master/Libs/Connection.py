import numpy as np
import base64
import datetime
import sys
import imutils
import cv2
import socket
import signal
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


proto_txt_path = 'model files\\face detection model\\deploy.prototxt'
model_path = 'model files\\face detection model\\res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)
mask_detector = load_model('model files\\face mask detection model\\mask_detector_1006.model')


def get_face_mask(frame):
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    faces = []
    bbox = []
    results = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faces.append(face)
            # print(len(faces))
            # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            bbox.append((startX, startY, endX, endY))
            # print(bbox)

    # print(len(faces))
    print("faces:", len(bbox))
    counter = 0

    for each, box in zip(faces, bbox):
        # print(each)
        results = mask_detector(each)

        for result in results:
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = result

            label = ""
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

    return frame


class Connection(object):

    def __init__(self, connections):
        print(connections)
        self.url = connections[1].decode("utf-8")
        self.gstream_flag = connections[2].decode("utf-8")
        print(self.url)
        self.socket = []
        self.socket.append(connections[0])
        self.connections = connections
        # signal.signal(signal.SIGINT, signal_handler)
        self.connect()
        pass



    def connect(self):
#        print(cv2.getBuildInformation())
        #if gstreamer is available then choose second video capture
        if(self.url == "0"):
            if(self.gstream_flag == "0"):
                self.connection = cv2.VideoCapture(0)
            else:
                self.connection = cv2.VideoCapture(0, cv2.CAP_GSTREAMER)
        else:
            if (self.gstream_flag == "0"):
                self.connection = cv2.VideoCapture(self.url)
            else:
                self.connection  = cv2.VideoCapture(self.url, cv2.CAP_GSTREAMER)
        return self.connection

    def addConnection(self, client):
        self.socket.append(client)

    def capture(self, opened_cameras):
        self.opened_cameras = opened_cameras
        while 1:
            try:
                ret, frame = self.connection.read()
                #ret = True
                if ret:
                    #cv2.imshow("test",frame)
                    #print(frame.shape)
                    #res, im_png = cv2.imencode('.png', frame)
                    #as per use case id we have to call python function and get output image
                    frame = get_face_mask(frame)

                    frame = cv2.resize(frame, (400, 400))
                    data = cv2.imencode('.jpg', frame)[1].tostring()
                    data = base64.b64encode(data)

                    if len(self.socket):
                       for c in self.socket:
                           self.send(c,data)
                       # print("send done one time ")
                       # break;
                    else:
                       self.connection.release()
                       del self.opened_cameras[self.connections[1]]
                       exit(0)
                else:
                    print("test read, reconnecting in 5 sec", ret)
                    cv2.waitKey(5000)
                    self.connect()
                    # self.connections[1].close()
            except KeyboardInterrupt:
                self.signal_handler()

    def send(self,c, data):
        try:
            c.send(data)
            c.send(b"END!") # send param to end loop in client
            #print("data len::::",data.)
        except socket.error:
            self.socket.remove(c)
