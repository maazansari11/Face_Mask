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
import glob
from database_entry_live_mask import entry_to_db_live

OUTPUT_FRAMES = 'Face_Mask'

try:
    if not os.path.isdir(OUTPUT_FRAMES):
        os.mkdir(OUTPUT_FRAMES)
except:
    print("could not make the directory")

proto_txt_path = 'model files\\face detection model\\deploy.prototxt'
model_path = 'model files\\face detection model\\res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

mask_detector = load_model('model files\\face mask detection model\\mask_detector_new.model')

# cap = cv2.VideoCapture('E:\\Github Projects\\FaceMaskDetection\\VID20200602154358.mp4')
# cap = cv2.VideoCapture('http://192.168.10.101:8080/videofeed?username=password=')
cap = cv2.VideoCapture(0)
startX = startY = endX = endY = 0
writer = None
frame_num = 0
violation_frame = 0
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
violation_came = False
camera_stop = False
timeCame = False
inserted = False
start_frame = 0
end_frame = 0
fps_avg = []
AVG = 0
fps_check = False
color = (0, 0, 0)
no_of_violation = 0
label = ""


def create_violation(start, vf, end, name_video, path_video, ip, user, port, password):
    img_arr = []
    global no_of_violation, violation_came
    violation_video = "video_violations" + os.sep + "YEAR " + time.strftime("%Y") + os.sep + "MONTH " + time.strftime(
        "%m") + os.sep + "DATE " + time.strftime("%d") + os.sep + "HOUR " + time.strftime(
        "%H") + os.sep + "violation" + str(no_of_violation) + ".mp4"
    # print(violation_video)
    # print("video_violations" + os.sep + "YEAR " + time.strftime("%Y") + os.sep + "MONTH " + time.strftime("%m") + os.sep + "DATE " + time.strftime("%d") + os.sep + "HOUR " + time.strftime("%H"))
    if not os.path.exists(os.getcwd() + os.sep + "video_violations" + os.sep + "YEAR " + time.strftime(
            "%Y") + os.sep + "MONTH " + time.strftime("%m") + os.sep + "DATE " + time.strftime(
        "%d") + os.sep + "HOUR " + time.strftime("%H")):
        # print("hi")
        os.makedirs(os.getcwd() + os.sep + "video_violations" + os.sep + "YEAR " + time.strftime(
            "%Y") + os.sep + "MONTH " + time.strftime("%m") + os.sep + "DATE " + time.strftime(
            "%d") + os.sep + "HOUR " + time.strftime("%H"))

    size = (0, 0)
    for fn in range(start, end):
            frame_video = OUTPUT_FRAMES + "\\frame_" + str(fn) + ".jpg"
            # print("fv", frame_video, " is writing into ", violation_video, "at FPS", AVG, type(AVG))
            # cv2.imshow("frame", cv2.imread(frame_video))
            # cv2.waitKey(0)
            frame_ = cv2.imread(frame_video)
            try:
                height, width, layers = frame_.shape
                size = (width, height)
            except:
                pass
            img_arr.append(frame_)

    violation = cv2.VideoWriter(violation_video, cv2.VideoWriter_fourcc(*'mp4v'), AVG, size)

    for i in range(len(img_arr)):
        violation.write(img_arr[i])
    violation.release()
    img_arr.clear()
    vf_frame = OUTPUT_FRAMES + "\\frame_" + str(vf) + ".jpg"
    vf_img = cv2.imread(vf_frame)
    vf_frame_path = "video_violations" + os.sep + "YEAR " + time.strftime("%Y") + os.sep + "MONTH " + time.strftime(
        "%m") + os.sep + "DATE " + time.strftime("%d") + os.sep + "HOUR " + time.strftime(
        "%H") + os.sep + "violation_frame" + str(no_of_violation) + ".jpg"
    cv2.imwrite(vf_frame_path, vf_img)
    no_of_violation += 1
    print("[INFO] Generated violation video.")

    path_violation_frame = vf_frame_path
    path_violation_video = violation_video
    cwd = os.getcwd() + os.sep
    entry_to_db_live(name_video, cwd + path_video,
                     cwd + path_violation_video, cwd + path_violation_frame, ip, user, port, password)

    for img in glob.glob("Face_Mask\\*.jpg"):
        os.remove(img)


def main_method(cap):
    global frame_num, total_frames, violation_came, AVG, start_frame, end_frame, label, violation_frame, color

    while True:
        create_name_video = "live_camera_recordings" + os.sep + time.strftime(
            "%Y") + os.sep + time.strftime("%m") + os.sep + time.strftime(
            "%d") + os.sep + time.strftime("%H") + os.sep + "Video_" + time.strftime("%H_%M") + ".mp4"
        print("[INFO] Live recording feed: ", create_name_video)
        if not os.path.exists("live_camera_recordings" + os.sep + time.strftime(
                "%Y") + os.sep + time.strftime("%m") + os.sep + time.strftime(
            "%d") + os.sep + time.strftime("%H")):
            os.makedirs("live_camera_recordings" + os.sep + time.strftime(
                "%Y") + os.sep + time.strftime("%m") + os.sep + time.strftime(
                "%d") + os.sep + time.strftime("%H"))

        total_frames = total_frames + 1
        ret, frame = cap.read()
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
        # print("faces:", len(bbox))
        counter = 0

        for each, box in zip(faces, bbox):
            # print(each)
            results = mask_detector(each)

            for result in results:
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = result

                if mask > withoutMask:
                    label = "Mask"
                    color = (0, 255, 0)
                    # frame_num += 1
                elif not violation_came and AVG > 0:
                    label = "No Mask"
                    color = (0, 0, 255)
                    violation_came = True
                    import math
                    # print("inside violation condition")
                    violation_frame = frame_num
                    start_frame = violation_frame - math.ceil(AVG * 3)
                    end_frame = violation_frame + math.floor(AVG * 4)
                    # frame_num += 1
                else:
                    label = "No Mask"
                    color = (0, 0, 255)

                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

                print(frame_num, start_frame, end_frame, violation_frame)
                if frame_num >= end_frame > 0 and start_frame > 0:
                    # print("if if")
                    create_violation(start_frame, violation_frame, end_frame,
                                     create_name_video.split("\\")[-1].split(".")[-2],
                                     create_name_video, "camera_ip", "camera_user", "camera_port",
                                     "camera_password")
                    print("[INFO] clearing frame numbers")
                    violation_came = False
                    violation_frame = 0
                    start_frame = 0
                    end_frame = 0
                # else:
                #     frame_num += 1

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
            if len(fps_avg) <= 5:
                fps = "{:.2f}".format(fps)
                # print(type(fps))
                fps_avg.append(float(fps))
            else:
                AVG = sum(fps_avg) // len(fps_avg)
                fps_check = True

        # print(AVG)
        out = cv2.VideoWriter(create_name_video, cv2.VideoWriter_fourcc(*'mp4v'), AVG, (600, 400))
        # print("counter", counter)
        cv2.imshow("Frame", frame)
        cv2.imwrite(OUTPUT_FRAMES + "\\frame_" + str(frame_num) + ".jpg", frame)
        frame_num += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break


main_method(cap)