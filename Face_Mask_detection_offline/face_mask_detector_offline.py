import glob
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime
from database_entry_live import entry_to_db_live

OUTPUT_FRAMES = 'Face_Mask'

try:
    if not os.path.isdir(OUTPUT_FRAMES):
        os.mkdir(OUTPUT_FRAMES)
except:
    print("could not make the directory")

proto_txt_path = 'model files\\face detection model\\deploy.prototxt'
model_path = 'model files\\face detection model\\res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

mask_detector = load_model('model files\\face mask detection model\\mask_detector_1006.model')

# cap = cv2.VideoCapture('E:\\Github Projects\\FaceMaskDetection\\VID20200602154358.mp4')
# cap = cv2.VideoCapture('http://192.168.10.101:8080/videofeed?username=password=')
cap = cv2.VideoCapture(0)
startX = startY = endX = endY = 0
writer = None
frame_num = 0
violation_came = False
violation_frame = 0
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0

start_frame = 0
end_frame = 0

offline_video_path = 'E:\\Github Projects\\AIComputerVision\\video\\mask.mp4'


def preprocess():
    if not os.path.exists('face_mask_input'):
        print("[INFO] Creating input Directory...")
        os.makedirs('face_mask_input')

    if not os.path.exists('face_mask_output'):
        print("[INFO] Creating output Directory...")
        os.makedirs('face_mask_output')

    print("[INFO] Removing previous images from input Directory...")
    for file in glob.glob('face_mask_input\\*.jpg'):
        os.remove(file)

    print("[INFO] Removing previous images from output Directory...")
    for file in glob.glob('face_mask_output\\*.jpg'):
        os.remove(file)


for i in range(glob.glob("face_mask_input\\*.jpg")):
    total_frames = total_frames + 1
    frame = cv2.imread(i)
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

    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    # if not violation_came:
    #     #  TODO: change condition of violation for actual violation
    #     if frame_num == 10:
    #         violation_came = True
    #         import math
    #         # print("inside violation condition")
    #         violation_frame = frame_num
    #         start_frame = violation_frame - math.ceil(fps * 2)
    #         end_frame = violation_frame + math.floor(fps * 3)
    #         frame_num += 1
    #     else:
    #         frame_num += 1
    # else:
    #     frame_num += 1
    #
    # if frame_num >= end_frame and start_frame > 0 and end_frame > 0:
    #     # print("if if")
    #     create_violation(start_frame, violation_frame, end_frame, video_name.split("\\")[-1].split(".")[-2], video_name.replace("\\", "\\\\"), camera_ip, camera_user, camera_port, camera_password)
    #     print("[INFO] clearing frame nnumbers")
    #     violation_came = False
    #     violation_frame = 0
    #     start_frame = 0
    #     end_frame = 0
    # else:
    #     print("if else")
    #     frame_number += 1

    # print("counter", counter)
    cv2.imshow("Frame", frame)
    cv2.imwrite("{FRAME_DIR}/frame_{num:05d}.jpg".format(FRAME_DIR=OUTPUT_FRAMES, num=frame_num), frame)
    frame_num += 1
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('face_mask_DEMO.avi', fourcc, 5,
                                 (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)

    # if len(faces) > 0:
    #     results = mask_detector.predict(faces)
    #     # print(results)
    #
    # for (face_box, result) in zip(bbox, results):
    #     (startX, startY, endX, endY) = face_box
    #     (mask, withoutMask) = result
    #
    # label = ""
    # if mask > withoutMask:
    #     label = "Mask"
    #     color = (0, 255, 0)
    # else:
    #     label = "No Mask"
    #     color = (0, 0, 255)
    #
    # cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    # cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
