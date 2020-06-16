import glob
import csv
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os

OUTPUT_FRAMES = 'face_mask_output'

proto_txt_path = 'model files\\face detection model\\deploy.prototxt'
model_path = 'model files\\face detection model\\res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

mask_detector = load_model('model files\\face mask detection model\\mask_detector_new.model')

startX = startY = endX = endY = violation_frame = frame_num = start_frame = end_frame = 0
violation_came = camera_stop = inserted = False
img_array = videos = []
writer = None


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


def create_video(frame_number, fps):
    global no_of_videos
    no_of_videos += 1
    print("[INFO] Creating Video...")
    frame = frame_number - (int(fps) * 2)
    video_range = frame_number + (int(fps) * 3)
    s = (0, 0)

    for i in range(video_range):
        file_name = "face_mask_output\\" + frame + ".jpg"
        img = cv2.imread(file_name)
        h, w, l = img.shape
        s = (w, h)
        img_array.append(img)
        frame += 1

    out_file = cv2.VideoWriter('face_mask_output' + str(no_of_videos) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, s)

    print("[INFO] Making videos from images...")
    for i in range(len(img_array)):
        out_file.write(img_array[i])
    out_file.release()
    print("[INFO] video created as face_mask_output" + str(no_of_videos) + ".mp4")


def parse_args():
    desc = 'Capture and display offline video'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--filepath', dest='filepath', type=str, help='Enter a file Path to pprocess.')
    args = parser.parse_args()
    return args


def show(frame, fps):
    global frame_num, inserted, camera_stop
    # frame = cv2.imread(frame)
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
            bbox.append((startX, startY, endX, endY))

    # print("faces:", len(bbox))

    for each, box in zip(faces, bbox):
        results = mask_detector(each)

        for result in results:
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = result

            label = ""
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
                inserted = False
            else:
                label = "No Mask"
                color = (0, 0, 255)
                if not inserted:
                    videos.append([frame_num - int(fps * 2), frame_num + int(fps * 3)])
                    inserted = True

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

    cv2.imshow("Frame", frame)
    cv2.imwrite("{FRAME_DIR}/frame_{num:05d}.jpg".format(FRAME_DIR=OUTPUT_FRAMES, num=frame_num), frame)
    frame_num += 1
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        camera_stop = True


preprocess()
argument = parse_args()

print("[INFO] Detecting from video...")
camera = str(argument.filepath)

print(camera)
video = cv2.VideoCapture(camera)
FPS = video.get(cv2.CAP_PROP_FPS)

while not camera_stop:
    ret, image = video.read()
    if not ret:
        print("[INFO] Done detecting.")
        break
    show(image, FPS)

video.release()
cv2.destroyAllWindows()

print("[INFO] Printing Vidoes List")
for i in videos:
    print(i)
    with open("videos_list.csv", "w+") as file:
        writer = csv.writer(file)
        writer.writerow([i])
