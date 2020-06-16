import cv2
import threading
import sys
import glob
import requests
import json
import argparse
import urllib3
import numpy as np
import os


urllib3.disable_warnings()
# print("Warning: Certificates not verified!")
camera_stop = False
camera_online = False
num = 0


def put_frame(image):
    global filename, num
    try:
        filename = "input\\frame"+num+".jpg"
        print(filename)
        cv2.imwrite(filename, image)
    except cv2.error:
        sys.exit()


def parse_args():
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--ip', dest='ip', type=str,
                        help='use IP CAM (remember to also set --uri)', default="195.229.90.110")
    parser.add_argument('--user', dest='user', type=str,
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default='admin')
    parser.add_argument('--password', dest='password', type=str,
                        help='latency in ms for RTSP [200]',
                        default='India12345')
    parser.add_argument('--port', dest='port', type=str,
                        help='use USB webcam (remember to also set --vid)', default="4444")
    args = parser.parse_args()
    return args


def show_image(image, video, ip, port):
    global camera_stop
    try:
        cv2.imshow(ip + ":" + port, image)
        if cv2.waitKey(1) == ord('q'):
            camera_stop = True
            video.release()
            cv2.destroyAllWindows()
            sys.exit(1)
    except cv2.error:
        sys.exit(1)


def detect_from_image(image):
    global filename
    global frame_number
    global inserted
    global violation_frame
    global start_frame, end_frame
    api_url = "https://10.150.20.65/visual-insights/api/dlapis/95771bd7-774e-466f-be7b-7d7a6967a6b9"
    cv2.imwrite("image.jpg", image)
    try:
        with open("image.jpg", 'rb') as f:
            s = requests.Session()
            r = s.post(api_url, files={'files': ("image.jpg", f)}, verify=False, timeout=10)

        if r.text is not None:
            data = json.loads(r.text)

        if data['result'] != 'fail':
            testdata = data["classified"]

            font = cv2.FONT_HERSHEY_SIMPLEX
            text1 = "You can press 'Q' to STOP the camra."
            textsize = cv2.getTextSize(text1, font, 1, 2)[0]
            textX = (image.shape[1] - textsize[0]) // 2
            cv2.putText(image, text1, (textX, 30), 2, 1, (255, 255, 255), 1)
            cv2.putText(image, text1, (textX, 370), 2, 1, (0, 0, 0), 1)

            for counter in range((len(testdata))):
                if testdata[counter].get('label') == 'camera':
                    SminX = int(testdata[counter].get('xmin'))
                    SminY = int(testdata[counter].get('ymin'))
                    SmaxX = int(testdata[counter].get('xmax'))
                    SmaxY = int(testdata[counter].get('ymax'))
                    cv2.rectangle(image, (SminX, SminY), (SmaxX, SmaxY), (0, 255, 0), 2)
                    cv2.putText(image, str(testdata[counter].get('label')), (SminX, SminY + 10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

                elif testdata[counter].get('label') == 'duct':
                    MminX = int(testdata[counter].get('xmin'))
                    MminY = int(testdata[counter].get('ymin'))
                    MmaxX = int(testdata[counter].get('xmax'))
                    MmaxY = int(testdata[counter].get('ymax'))
                    cv2.rectangle(image, (MminX, MminY), (MmaxX, MmaxY), (0, 255, 0), 2)
                    cv2.putText(image, str(testdata[counter].get('label')), (MminX, MminY + 10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
        else:
            print("[INFO] Failed from PowerAI.")
    except:
        pass


def main_method(camera, ip, port):
    global timeCame
    global frame_number
    global camera_online
    video = cv2.VideoCapture(camera, cv2.CAP_GSTREAMER)
    for frames in glob.glob("input\\*.jpg"):
        os.remove(frames)

    print("remove")
    while True:
        ret, frame = video.read()
        # print(ret)
        if not camera_online:
            white = (255, 255, 255)
            blank_image = np.zeros((400, 600, 3), np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text2 = "Waiting for camera to respond"
            textsize2 = cv2.getTextSize(text2, font, 1, 2)[0]
            text2X = (blank_image.shape[1] - textsize2[0]) // 2
            text2Y = (blank_image.shape[0] + textsize2[1]) // 2
            cv2.putText(blank_image, text2, (text2X, text2Y), 2, 1, white, 1)
            cv2.imwrite("input\\initial_image.jpg", blank_image)
            cv2.imshow("Description Page", blank_image)
            cv2.waitKey(1)
        if not camera_stop:
            if not ret:
                import time
                video = cv2.VideoCapture(camera, cv2.CAP_GSTREAMER)
                continue
        if not ret:
            break
        cv2.destroyWindow("Description Page")
        camera_online = True
        frame = cv2.resize(frame, (600, 400))
        t1 = threading.Thread(target=put_frame(frame, ), )
        t1.start()
        t3 = threading.Thread(target=detect_from_image(frame), )
        t3.start()
        t2 = threading.Thread(target=show_image(frame, video, ip, port), )
        t2.start()

argument = parse_args()

camera = 'rtspsrc location=rtsp://' + str(argument.user) + ':' + str(argument.password) + '@' + str(
    argument.ip) + ':' + str(
    argument.port) + ' latency=2000 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink '

# white = (255, 255, 255)
# blank_image = np.zeros((400,600,3), np.uint8)
# font = cv2.FONT_HERSHEY_SIMPLEX
# text2 = "Waiting for camera to respond"
# textsize2 = cv2.getTextSize(text2, font, 1, 2)[0]
# text2X = (blank_image.shape[1] - textsize2[0]) // 2
# text2Y = (blank_image.shape[0] + textsize2[1]) // 2
# cv2.putText(blank_image, text2, (text2X, text2Y), 2, 1, white, 1)
# cv2.imshow("Description Page", blank_image)
# cv2.waitKey(1)

try:
    main_method(camera, str(argument.ip), str(argument.port))

except:
    print("Thank You.")
