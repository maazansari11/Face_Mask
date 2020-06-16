from database_entry import entry_to_db
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from _datetime import datetime
import argparse
import imutils
import glob
import cv2
import os

OUTPUT_FRAMES = 'face_mask_output'

proto_txt_path = 'model files\\face detection model\\deploy.prototxt'
model_path = 'model files\\face detection model\\res10_300x300_ssd_iter_140000.caffemodel'
mask_detector = load_model('model files\\face mask detection model\\mask_detector_new.model')
face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

startX = startY = endX = endY = 0
violation_frame = 0
frame_num = 0
start_frame = 0
end_frame = 0
violation_came = False
camera_stop = False
inserted = False
img_array = []
videos = []
original_list = []
writer = None


def preprocess():
    if not os.path.exists('face_mask_input'):
        os.makedirs('face_mask_input')
    if not os.path.exists('face_mask_output'):
        os.makedirs('face_mask_output')
    for imagepath1 in glob.glob('face_mask_input\\*.jpg'):
        os.remove(imagepath1)
    for imagepath2 in glob.glob('face_mask_output\\*.jpg'):
        os.remove(imagepath2)


def create_video(video_name, video_violation_list, fps):
    print("[INFO] creating video")
    no_of_videos = 0
    print(video_name, video_violation_list)

    for v in video_violation_list:
        s = (0, 0)
        start_frame, end_frame = v[0], v[1]
        print("sf", start_frame, "ef", end_frame)
        violation_no = "violation_" + str(no_of_videos)

        for fn in range(start_frame, end_frame):
            print("fn", fn)
            if start_frame <= end_frame:
                file_name = "face_mask_output" + os.sep + str(fn) + ".jpg"
                print("filename", file_name)
                img = cv2.imread(file_name)
                try:
                    h, w, l = img.shape
                    s = (w, h)
                    img_array.append(img)
                except AttributeError:
                    break
            else:
                break

        if not os.path.exists(
                "processed_video" + os.sep + video_name.split(".")[-2].split("\\")[
                    -1] + os.sep + datetime.now().strftime(
                    "%Y") + os.sep + datetime.now().strftime("%m") + os.sep + datetime.now().strftime(
                    "%d") + os.sep + datetime.now().strftime("%H") + os.sep + violation_no):
            os.makedirs(
                "processed_video" + os.sep + video_name.split(".")[-2].split("\\")[
                    -1] + os.sep + datetime.now().strftime(
                    "%Y") + os.sep + datetime.now().strftime("%m") + os.sep + datetime.now().strftime(
                    "%d") + os.sep + datetime.now().strftime("%H") + os.sep + violation_no)

        violation_video = video_name.split(".")[-2].split("\\")[-1] + "_violation" + str(no_of_videos) + '.mp4'
        vv_path = "processed_video" + os.sep + video_name.split(".")[-2].split("\\")[
            -1] + os.sep + datetime.now().strftime(
            "%Y") + os.sep + datetime.now().strftime("%m") + os.sep + datetime.now().strftime(
            "%d") + os.sep + datetime.now().strftime("%H") + os.sep + violation_no + os.sep + violation_video
        print("Video Violtion path", vv_path)
        out_file = cv2.VideoWriter(vv_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, s)

        for j in range(len(img_array)):
            out_file.write(img_array[j])

        vf_name = video_name.split(".")[-2].split("\\")[-1] + "_violation_frame_" + str(no_of_videos) + ".jpg"
        vf_path = "processed_video" + os.sep + video_name.split(".")[-2].split("\\")[
            -1] + os.sep + datetime.now().strftime(
            "%Y") + os.sep + datetime.now().strftime("%m") + os.sep + datetime.now().strftime(
            "%d") + os.sep + datetime.now().strftime("%H") + os.sep + violation_no + os.sep + vf_name
        print(vf_path)
        cv2.imwrite(vf_path, img_array[(len(img_array) // 2)])

        # TODO: Make entry in mysql datbase
        print("============================================================================================================")
        cwd = os.getcwd() + "\\"
        print(cwd.replace("\\", "\\\\") + "video_for_process\\\\")
        print(video_name)
        print(cwd.replace("\\", "\\\\") + "video_for_process\\\\" + video_name)
        print(cwd.replace("\\", "\\\\") + vv_path.replace("\\", "\\\\"))
        print(cwd.replace("\\", "\\\\") + vf_path.replace("\\", "\\\\"))
        print("============================================================================================================")
        entry_to_db(cwd.replace("\\", "\\\\") + "video_for_process\\\\", video_name, cwd.replace("\\", "\\\\") + "video_for_process\\\\" + video_name, cwd.replace("\\", "\\\\") + vv_path.replace("\\", "\\\\"), cwd.replace("\\", "\\\\") + vf_path.replace("\\", "\\\\"))

        out_file.release()
        img_array.clear()
        no_of_videos += 1


def parse_args():
    desc = 'Capture and display offline video'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--filepath', dest='filepath', type=str, help='Enter a file Path to pprocess.')
    args = parser.parse_args()
    return args


def show(frame, fps):
    global frame_num, inserted, camera_stop
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
                    if frame_num - int(fps * 2) < 0:
                        videos.append([0, frame_num + int(fps * 3)])
                    else:
                        videos.append([frame_num - int(fps * 2), frame_num + int(fps * 3)])
                    inserted = True

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

    cv2.imshow("Frame", frame)
    cv2.imwrite("face_mask_output\\" + str(frame_num) + ".jpg", frame)
    frame_num += 1
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        camera_stop = True


def list_slicing(video_list, original_list):
    if not video_list == []:
        start = video_list[0][0]
        end = video_list[0][1]
        original_list.append([start, end])
        video_list.pop(0)
        counter = 0

        for i in range(len(video_list)):
            i += counter
            if start < video_list[i][0] < end:
                video_list.remove(video_list[i])
                counter -= 1

        list_slicing(video_list, original_list)


def main(video_path):
    print("[INFO] in main with " + video_path)
    video = cv2.VideoCapture(video_path)
    FPS = video.get(cv2.CAP_PROP_FPS)
    while not camera_stop:
        ret, image = video.read()
        if not ret:
            print("[INFO] Done detecting.")
            print("videos", videos, original_list)
            list_slicing(videos, original_list)
            # Writing in csv
            # print("[INFO] Writing Vidoes List in csv")
            # for i in original_list:
            #     print(i)
            #     with open("videos_list.csv", "a+") as file:
            #         writer = csv.writer(file)
            #         writer.writerow([i])
            #         end writing
            create_video(video_path.split("\\")[-1], original_list, FPS)
            break
        show(image, FPS)
    video.release()
    cv2.destroyAllWindows()


preprocess()
# main("processed_video\\Video2_20sec.mp4")    # TODO: Remove this line in production
