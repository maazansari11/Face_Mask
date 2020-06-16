import os
import json
import glob
import requests
import cv2
import argparse
import urllib3
import csv

urllib3.disable_warnings()
print("[INFO] Warning: Certificates not verified!")

num = 0
img_array = []
size = []
no_of_videos = 0
videos = []
inserted = False
camera_stop = False
# api_url = "https://10.150.20.65/powerai-vision/api/dlapis/d3d5e84e-7b8a-42d5-acd7-30899dce3aa7"
api_url = "https://195.229.90.114/powerai-vision/api/dlapis/12b51a15-41f8-47a0-bde7-5b73b786a5f7"


def preprocess():
    if not os.path.exists('no_photo_zone_input'):
        print("[INFO] Creating input Directory...")
        os.makedirs('no_photo_zone_input')

    if not os.path.exists('no_photo_zone_output'):
        print("[INFO] Creating output Directory...")
        os.makedirs('no_photo_zone_output')

    print("[INFO] Removing previous images from input Directory...")
    for file in glob.glob('no_photo_zone_input\\*.jpg'):
        os.remove(file)

    print("[INFO] Removing previous images from output Directory...")
    for file in glob.glob('no_photo_zone_output\\*.jpg'):
        os.remove(file)


def create_video(frame_number, fps):
    global no_of_videos
    no_of_videos += 1
    print("[INFO] Creating Video...")
    frame = frame_number - (int(fps) * 2)
    video_range = frame_number + (int(fps) * 3)
    s = (0, 0)

    for i in range(video_range):
        file_name = "no_photo_zone_output\\" + frame + ".jpg"
        img = cv2.imread(file_name)
        h, w, l = img.shape
        s = (w, h)
        img_array.append(img)
        frame += 1

    out_file = cv2.VideoWriter('no_photo_zone_output' + str(no_of_videos) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, s)

    print("[INFO] Making videos from images...")
    for i in range(len(img_array)):
        out_file.write(img_array[i])
    out_file.release()
    print("[INFO] video created as no_photo_zone_output" + str(no_of_videos) + ".mp4")


def parse_args():
    desc = 'Capture and display offline video'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--filepath', dest='filepath', type=str, help='Enter a file Path to pprocess.')
    args = parser.parse_args()
    return args


def show(frame, fps):
    global num, camera_stop, inserted
    data = {}
    SminX, SminY, SmaxX, SmaxY = 0, 0, 0, 0
    MminX, MminY, MmaxX, MmaxY = -100, -100, -100, -100
    cv2.imwrite("image.jpg", frame)
    with open("image.jpg", 'rb') as f:
        s = requests.Session()
        r = s.post(api_url, files={'files': ("image.jpg", f)}, verify=False)

    if r.text is not None:
        data = json.loads(r.text)

    if data['result'] != 'fail':
        testdata = data["classified"]

        for counter in range((len(testdata))):
            if testdata[counter].get('label') == 'system':
                SminX = int(testdata[counter].get('xmin'))
                SminY = int(testdata[counter].get('ymin'))
                SmaxX = int(testdata[counter].get('xmax'))
                SmaxY = int(testdata[counter].get('ymax'))
                # cv2.rectangle(image, (SminX, SminY), (SmaxX, SmaxY), (0, 255, 0), 2)
                # cv2.putText(image, str(testdata[counter].get('label')), (SminX, SminY + 10),
                # cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

            elif testdata[counter].get('label') == 'mobile':
                MminX = int(testdata[counter].get('xmin'))
                MminY = int(testdata[counter].get('ymin'))
                MmaxX = int(testdata[counter].get('xmax'))
                MmaxY = int(testdata[counter].get('ymax'))
                # cv2.rectangle(image, (MminX, MminY), (MmaxX, MmaxY), (0, 255, 0), 2)
                # cv2.putText(image,
                # str(testdata[counter].get('label')), (MminX, MminY + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,
                # 0, 0), 1)

        if MminY >= SminY - 30 and MmaxY <= SmaxY - 10:
            cv2.putText(image, "Taking photo", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255), 2)
            if not inserted:
                videos.append({"start_frame_number": num - int(fps * 2), "end_frame_number": num + int(fps * 3)})
                inserted = True
        else:
            inserted = False

        cv2.imshow('image', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            camera_stop = True
        cv2.imwrite("no_photo_zone_output\\{num:5d}.jpg".format(num=num), image)
        num += 1


preprocess()
argument = parse_args()

print("[INFO] Detecting from video...")
camera = str(argument.filepath)

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
